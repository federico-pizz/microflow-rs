//! ONNX model parser for MicroFlow.
//!
//! Supports **QDQ-quantized** ONNX models (produced by ONNX Runtime's `quantize_static` or
//! PyTorch's INT8 export path).  In QDQ format every compute operator is wrapped by
//! `QuantizeLinear` / `DequantizeLinear` pairs that carry the per-tensor (or per-channel)
//! scale and zero_point; the macro folds those away and feeds the extracted quantisation
//! parameters directly into the generated code.
//!
//! Float-only models (no QDQ annotations) are also supported: weights are quantised at
//! compile time using symmetric per-tensor quantisation; activation scales default to 1/128
//! with zero_point = 0.  For best accuracy use a calibrated quantised ONNX model.

use std::collections::HashMap;

use nalgebra::DMatrix;
use prost::Message;
use proc_macro2::TokenStream as TokenStream2;
use proc_macro_error::abort_call_site;
use quote::{format_ident, quote, ToTokens};
use simba::scalar::SupersetOf;

use crate::buffer::{TokenBuffer2D, TokenBuffer4D};

// ─── Minimal ONNX protobuf message definitions ────────────────────────────────

#[derive(Clone, PartialEq, Message)]
pub struct OnnxModel {
    #[prost(int64, tag = "1")]
    pub ir_version: i64,
    #[prost(message, optional, tag = "7")]
    pub graph: Option<OnnxGraph>,
}

#[derive(Clone, PartialEq, Message)]
pub struct OnnxGraph {
    #[prost(message, repeated, tag = "1")]
    pub node: Vec<OnnxNode>,
    #[prost(string, tag = "2")]
    pub name: String,
    #[prost(message, repeated, tag = "5")]
    pub initializer: Vec<OnnxTensor>,
    #[prost(message, repeated, tag = "11")]
    pub input: Vec<OnnxValueInfo>,
    #[prost(message, repeated, tag = "12")]
    pub output: Vec<OnnxValueInfo>,
    #[prost(message, repeated, tag = "13")]
    pub value_info: Vec<OnnxValueInfo>,
}

#[derive(Clone, PartialEq, Message)]
pub struct OnnxNode {
    #[prost(string, repeated, tag = "1")]
    pub input: Vec<String>,
    #[prost(string, repeated, tag = "2")]
    pub output: Vec<String>,
    #[prost(string, tag = "3")]
    pub name: String,
    #[prost(string, tag = "4")]
    pub op_type: String,
    #[prost(string, tag = "7")]
    pub domain: String,
    #[prost(message, repeated, tag = "5")]
    pub attribute: Vec<OnnxAttr>,
}

#[derive(Clone, PartialEq, Message)]
pub struct OnnxAttr {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(int32, tag = "20")]
    pub r#type: i32,
    #[prost(float, tag = "4")]
    pub f: f32,
    #[prost(int64, tag = "3")]
    pub i: i64,
    #[prost(bytes = "vec", tag = "6")]
    pub s: Vec<u8>,
    #[prost(message, optional, tag = "5")]
    pub t: Option<OnnxTensor>,
    #[prost(float, repeated, tag = "7")]
    pub floats: Vec<f32>,
    #[prost(int64, repeated, tag = "8")]
    pub ints: Vec<i64>,
}

/// ONNX data_type constants used in this file.
const DT_FLOAT: i32 = 1;
// DT_UINT8 = 2 (reserved for future use)
const DT_INT8: i32 = 3;
const DT_INT32: i32 = 6;

#[derive(Clone, PartialEq, Message)]
pub struct OnnxTensor {
    #[prost(int64, repeated, tag = "1")]
    pub dims: Vec<i64>,
    #[prost(int32, tag = "2")]
    pub data_type: i32,
    #[prost(string, tag = "8")]
    pub name: String,
    /// float_data is packed=false so every value occupies its own record.
    #[prost(float, repeated, packed = "false", tag = "4")]
    pub float_data: Vec<f32>,
    #[prost(int32, repeated, packed = "false", tag = "5")]
    pub int32_data: Vec<i32>,
    #[prost(int64, repeated, packed = "false", tag = "7")]
    pub int64_data: Vec<i64>,
    #[prost(bytes = "vec", tag = "9")]
    pub raw_data: Vec<u8>,
}

#[derive(Clone, PartialEq, Message)]
pub struct OnnxValueInfo {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(message, optional, tag = "2")]
    pub r#type: Option<OnnxType>,
}

#[derive(Clone, PartialEq, Message)]
pub struct OnnxType {
    #[prost(message, optional, tag = "1")]
    pub tensor_type: Option<OnnxTypeTensor>,
}

#[derive(Clone, PartialEq, Message)]
pub struct OnnxTypeTensor {
    #[prost(int32, tag = "1")]
    pub elem_type: i32,
    #[prost(message, optional, tag = "2")]
    pub shape: Option<OnnxShape>,
}

#[derive(Clone, PartialEq, Message)]
pub struct OnnxShape {
    #[prost(message, repeated, tag = "1")]
    pub dim: Vec<OnnxDim>,
}

#[derive(Clone, PartialEq, Message)]
pub struct OnnxDim {
    #[prost(int64, tag = "1")]
    pub dim_value: i64,
    #[prost(string, tag = "2")]
    pub dim_param: String,
}

// ─── Quantisation parameters ──────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub(crate) struct Quant {
    pub scale: Vec<f32>,
    pub zero_point: Vec<i8>,
}

impl Quant {
    fn unit() -> Self {
        Self {
            scale: vec![1.0 / 128.0],
            zero_point: vec![0],
        }
    }
}

// ─── Tensor data helpers ───────────────────────────────────────────────────────

fn float_data(t: &OnnxTensor) -> Vec<f32> {
    if !t.raw_data.is_empty() {
        t.raw_data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    } else {
        t.float_data.clone()
    }
}

fn i8_data(t: &OnnxTensor) -> Vec<i8> {
    if !t.raw_data.is_empty() {
        t.raw_data.iter().map(|&b| b as i8).collect()
    } else {
        t.int32_data.iter().map(|&v| v as i8).collect()
    }
}

fn i32_data(t: &OnnxTensor) -> Vec<i32> {
    if !t.raw_data.is_empty() {
        t.raw_data
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    } else {
        t.int32_data.clone()
    }
}

fn tensor_shape(t: &OnnxTensor) -> Vec<usize> {
    t.dims.iter().map(|&d| d as usize).collect()
}

fn value_info_shape(vi: &OnnxValueInfo) -> Option<Vec<usize>> {
    vi.r#type
        .as_ref()?
        .tensor_type
        .as_ref()?
        .shape
        .as_ref()
        .map(|s| {
            s.dim
                .iter()
                .map(|d| if d.dim_value > 0 { d.dim_value as usize } else { 1 })
                .collect()
        })
}

/// Symmetric per-tensor quantisation: scale = max_abs / 127, zero_point = 0.
fn quant_sym(data: &[f32]) -> (Vec<i8>, f32) {
    let max_abs = data.iter().copied().fold(0f32, |m, v| m.max(v.abs()));
    let scale = if max_abs == 0.0 { 1.0 / 127.0 } else { max_abs / 127.0 };
    let q = data
        .iter()
        .map(|&v| (v / scale).round().clamp(-128.0, 127.0) as i8)
        .collect();
    (q, scale)
}

fn attr_i(attrs: &[OnnxAttr], name: &str) -> Option<i64> {
    attrs.iter().find(|a| a.name == name).map(|a| a.i)
}

fn attr_ints<'a>(attrs: &'a [OnnxAttr], name: &str) -> Option<&'a [i64]> {
    attrs.iter().find(|a| a.name == name).map(|a| a.ints.as_slice())
}

fn attr_s(attrs: &[OnnxAttr], name: &str) -> Option<String> {
    attrs
        .iter()
        .find(|a| a.name == name)
        .map(|a| String::from_utf8_lossy(&a.s).to_string())
}

// ─── Public output type ────────────────────────────────────────────────────────

/// Output of the ONNX parser: drives the same code-generation as the TFLite path.
pub(crate) struct ParsedOnnx {
    pub input_shape: Vec<usize>,
    pub input_quant: Quant,
    pub output_shape: Vec<usize>,
    /// Operator token streams in execution order (may include clone statements for branches).
    pub layers: Vec<TokenStream2>,
}

// ─── Main entry point ─────────────────────────────────────────────────────────

pub(crate) fn parse_onnx(buf: &[u8]) -> ParsedOnnx {
    let model = OnnxModel::decode(buf)
        .unwrap_or_else(|e| abort_call_site!("failed to parse ONNX model: {}", e));
    let graph = model
        .graph
        .unwrap_or_else(|| abort_call_site!("ONNX model has no graph"));

    // ── Build maps ───────────────────────────────────────────────────────────
    let inits: HashMap<String, OnnxTensor> = graph
        .initializer
        .iter()
        .map(|t| (t.name.clone(), t.clone()))
        .collect();

    let mut shape_map: HashMap<String, Vec<usize>> = HashMap::new();
    for t in &graph.initializer {
        shape_map.insert(t.name.clone(), tensor_shape(t));
    }
    for vi in graph.input.iter().chain(graph.output.iter()).chain(graph.value_info.iter()) {
        if let Some(s) = value_info_shape(vi) {
            shape_map.insert(vi.name.clone(), s);
        }
    }

    // ── QDQ folding: build quant_map ─────────────────────────────────────────
    // Maps tensor_name → Quant (scale + zero_point)
    let mut quant_map: HashMap<String, Quant> = HashMap::new();
    // Maps the "float-domain" output of DequantizeLinear → its int8 source name
    let mut dq_src_map: HashMap<String, String> = HashMap::new();

    for node in &graph.node {
        match node.op_type.as_str() {
            "QuantizeLinear" if node.input.len() >= 2 && !node.output.is_empty() => {
                let scale = read_float_init(&node.input[1], &inits);
                let zp = if node.input.len() >= 3 {
                    read_i8_init(&node.input[2], &inits)
                } else {
                    vec![0i8]
                };
                quant_map.insert(node.output[0].clone(), Quant { scale, zero_point: zp });
            }
            "DequantizeLinear" if node.input.len() >= 2 && !node.output.is_empty() => {
                let x = &node.input[0];
                let scale = read_float_init(&node.input[1], &inits);
                let zp = if node.input.len() >= 3 {
                    read_i8_init(&node.input[2], &inits)
                } else {
                    vec![0i8]
                };
                let q = Quant { scale, zero_point: zp };
                quant_map.insert(x.clone(), q.clone());
                quant_map.insert(node.output[0].clone(), q);
                dq_src_map.insert(node.output[0].clone(), x.clone());
            }
            _ => {}
        }
    }

    // ── Graph I/O ────────────────────────────────────────────────────────────
    let graph_inputs: Vec<_> = graph.input.iter().filter(|vi| !inits.contains_key(&vi.name)).collect();
    if graph_inputs.is_empty() {
        abort_call_site!("ONNX graph has no non-initializer inputs");
    }
    let graph_in_name = &graph_inputs[0].name;
    let graph_out_name = graph.output.first().map(|vi| &vi.name)
        .unwrap_or_else(|| abort_call_site!("ONNX graph has no outputs"));

    let mut input_shape = shape_map.get(graph_in_name.as_str()).cloned()
        .unwrap_or_else(|| abort_call_site!("cannot determine input shape"));
    if input_shape.len() == 1 { input_shape.insert(0, 1); }
    let input_quant = quant_map.get(graph_in_name.as_str()).cloned().unwrap_or_else(Quant::unit);

    let mut output_shape = shape_map.get(graph_out_name.as_str()).cloned()
        .unwrap_or_else(|| abort_call_site!("cannot determine output shape"));
    if output_shape.len() == 1 { output_shape.insert(0, 1); }

    // ── Phase 1a: find tensors that need saving for Concat/Add/Mul ───────────
    let channel_split_secondaries: std::collections::HashSet<String> = graph.node.iter()
        .filter(|n| n.op_type == "Split")
        .flat_map(|n| n.output.iter().skip(1).cloned())
        .collect();

    let mut to_save: std::collections::HashSet<String> = std::collections::HashSet::new();
    for node in &graph.node {
        if matches!(node.op_type.as_str(), "Concat" | "Add" | "Mul") {
            for inp in &node.input {
                if !channel_split_secondaries.contains(inp) {
                    to_save.insert(inp.clone());
                }
            }
        }
    }

    // ── Phase 1b: fused-activation map ───────────────────────────────────────
    // Maps a compute-op's output name → the activation that immediately follows it.
    // Used so Conv/Gemm can emit FusedActivation::Relu/Relu6 instead of None.
    let mut fused_activation_map: HashMap<String, &OnnxNode> = HashMap::new();
    for (i, node) in graph.node.iter().enumerate() {
        if matches!(node.op_type.as_str(),
            "Conv" | "ConvInteger" | "ConvTranspose" | "Gemm" | "MatMul"
            | "Add" | "Mul" | "AveragePool" | "MaxPool"
        ) {
            if let Some(next) = graph.node.get(i + 1) {
                if matches!(next.op_type.as_str(), "Relu" | "Clip") {
                    if let Some(out) = node.output.first() {
                        fused_activation_map.insert(out.clone(), next);
                    }
                }
            }
        }
    }

    // ── Phase 2: generate layers ──────────────────────────────────────────────
    let mut current_name = graph_in_name.clone();
    let mut layers: Vec<TokenStream2> = Vec::new();
    let mut op_idx = 0usize;
    let mut skip_next = false;

    for node in &graph.node {
        // A Relu/Clip that was fused into the preceding op is already handled.
        if skip_next {
            skip_next = false;
            if !node.output.is_empty() {
                current_name = node.output[0].clone();
            }
            continue;
        }

        match node.op_type.as_str() {
            "QuantizeLinear" | "DequantizeLinear" => {
                if !node.output.is_empty() {
                    current_name = node.output[0].clone();
                }
                continue;
            }
            "Relu" | "Clip" => {
                // Was not fused into the preceding op. Fall through to main dispatch
                // so qin/qout are resolved and the activation is emitted explicitly.
            }
            _ => {}
        }

        let out_name = node.output.first().cloned().unwrap_or_default();
        let qin = quant_of(&current_name, &quant_map, &dq_src_map);
        let qout = quant_of(&out_name, &quant_map, &dq_src_map);

        // Determine fused activation for ops that support it.
        let fused_act = if let Some(act_node) = fused_activation_map.get(&out_name) {
            skip_next = true;
            fused_activation_from_node(act_node)
        } else {
            quote!(microflow::activation::FusedActivation::None)
        };

        let layer_ts = match node.op_type.as_str() {
            "Conv" | "ConvInteger" => gen_conv(
                node, &inits, &shape_map, &quant_map, &dq_src_map,
                &current_name, &out_name, op_idx, &fused_act,
            ),
            "ConvTranspose" => gen_conv_transpose(
                node, &inits, &shape_map, &quant_map, &dq_src_map,
                &current_name, &out_name, op_idx, &fused_act,
            ),
            "Gemm" | "MatMul" => gen_gemm(
                node, &inits, &shape_map, &quant_map, &dq_src_map,
                &current_name, &out_name, op_idx, &fused_act,
            ),
            "AveragePool" | "GlobalAveragePool" => gen_avg_pool(
                node, &shape_map, qin, qout, &current_name, &out_name, &fused_act,
            ),
            "MaxPool" => gen_max_pool(node, &shape_map, qin, qout, &out_name, &fused_act),
            "Softmax" => gen_softmax(qout, &out_name, &shape_map),
            "Reshape" | "Flatten" => gen_reshape(&out_name, &shape_map),
            "Resize" => gen_resize(node, &shape_map, qin, qout, &out_name),
            "Concat" => gen_concat(
                node, &quant_map, &dq_src_map, &shape_map,
                &current_name, &out_name,
            ),
            "Add" => gen_add(node, &quant_map, &dq_src_map, &shape_map, &current_name, &out_name, &fused_act),
            "Mul" => gen_mul(node, &quant_map, &dq_src_map, &shape_map, &current_name, &out_name, &fused_act),
            "Relu" | "Clip" => gen_standalone_activation(node, qout, &out_name, &shape_map),
            "Sigmoid" => gen_sigmoid(qout, &out_name, &shape_map),
            "Tanh" => gen_tanh(qout, &out_name, &shape_map),
            "HardSwish" => gen_hard_swish(qout, &out_name, &shape_map),
            "LeakyRelu" => gen_leaky_relu(node, qout, &out_name, &shape_map),
            "ReduceMean" => gen_reduce_mean(node, &shape_map, qin, qout, &out_name),
            "Split" => {
                let ts = gen_channel_split(node, &quant_map, &shape_map, &current_name);
                layers.push(ts);
                current_name = node.output[0].clone();
                op_idx += 1;
                continue;
            }
            unknown => abort_call_site!("unsupported ONNX operator: {}", unknown),
        };

        layers.push(layer_ts);

        // Save a clone if this output is consumed by a later 2-input op.
        if to_save.contains(&out_name) {
            let var = name_to_ident(&out_name);
            layers.push(quote! { let #var = input.clone(); });
        }

        current_name = out_name;
        op_idx += 1;
    }

    ParsedOnnx { input_shape, input_quant, output_shape, layers }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn read_float_init(name: &str, inits: &HashMap<String, OnnxTensor>) -> Vec<f32> {
    inits.get(name).map(float_data).unwrap_or_else(|| vec![1.0])
}

fn read_i8_init(name: &str, inits: &HashMap<String, OnnxTensor>) -> Vec<i8> {
    inits.get(name).map(i8_data).unwrap_or_else(|| vec![0])
}

/// Resolve quantisation for a tensor, following DQ sources if needed.
fn quant_of<'a>(
    name: &str,
    quant_map: &'a HashMap<String, Quant>,
    dq_src: &HashMap<String, String>,
) -> &'a Quant {
    static DEFAULT: std::sync::OnceLock<Quant> = std::sync::OnceLock::new();
    let resolved = dq_src.get(name).map(|s| s.as_str()).unwrap_or(name);
    quant_map
        .get(resolved)
        .or_else(|| quant_map.get(name))
        .unwrap_or_else(|| DEFAULT.get_or_init(Quant::unit))
}

fn get_shape<'a>(name: &str, map: &'a HashMap<String, Vec<usize>>) -> &'a Vec<usize> {
    map.get(name)
        .unwrap_or_else(|| abort_call_site!("no shape for tensor '{}'", name))
}

/// Turn an ONNX tensor name into a Rust identifier (e.g. for `tensor_name_foo`).
fn name_to_ident(name: &str) -> proc_macro2::Ident {
    let sanitized: String = name
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect();
    format_ident!("tensor_{}", sanitized)
}

// ─── Code-generation for each operator ───────────────────────────────────────

fn gen_conv(
    node: &OnnxNode,
    inits: &HashMap<String, OnnxTensor>,
    shape_map: &HashMap<String, Vec<usize>>,
    quant_map: &HashMap<String, Quant>,
    dq_src: &HashMap<String, String>,
    current_name: &str,
    out_name: &str,
    idx: usize,
    fused_act: &TokenStream2,
) -> TokenStream2 {
    let groups = attr_i(&node.attribute, "group").unwrap_or(1) as usize;
    let strides = attr_ints(&node.attribute, "strides")
        .map(|s| (s[0] as usize, s[1] as usize))
        .unwrap_or((1, 1));
    let pads = attr_ints(&node.attribute, "pads")
        .map(|p| p.to_vec())
        .unwrap_or(vec![0i64; 4]);
    let padding_ts = if pads.iter().any(|&p| p != 0) {
        quote!(microflow::tensor::TensorViewPadding::Same)
    } else {
        quote!(microflow::tensor::TensorViewPadding::Valid)
    };
    let (stride_h, stride_w) = strides;

    let w_name = &node.input[1];
    let w_src = dq_src.get(w_name).map(|s| s.as_str()).unwrap_or(w_name);
    let w_init = inits.get(w_src)
        .unwrap_or_else(|| abort_call_site!("Conv weight '{}' not found", w_src));
    let w_shape = tensor_shape(w_init); // [out_ch, in_ch/g, kH, kW]
    let (out_ch, in_ch_per_g, kh, kw) = (w_shape[0], w_shape[1], w_shape[2], w_shape[3]);

    let in_shape = get_shape(current_name, shape_map);
    let out_shape = get_shape(out_name, shape_map);
    let input_channels = *in_shape.last().unwrap_or(&1);
    let is_depthwise = groups > 1 && groups == input_channels;

    let qw = quant_of(w_src, quant_map, dq_src);
    let qin = quant_of(current_name, quant_map, dq_src);
    let qout = quant_of(out_name, quant_map, dq_src);

    // Weight data as i8
    let w_i8: Vec<i8> = match w_init.data_type {
        DT_INT8 => i8_data(w_init),
        DT_FLOAT => {
            let floats = float_data(w_init);
            let (q, _) = quant_sym(&floats);
            q
        }
        dt => abort_call_site!("unsupported Conv weight dtype {}", dt),
    };

    // Bias
    let (bias_i32, bias_scale) = read_bias(node, inits, dq_src, qin, qw, out_ch);

    if is_depthwise {
        // ONNX depthwise layout: [out_ch, 1, kH, kW] → microflow [1, kH, kW, out_ch]
        let w_buf: Vec<DMatrix<Vec<i8>>> = vec![
            DMatrix::from_fn(kh, kw, |h, w_c| {
                (0..out_ch).map(|oc| w_i8[oc * (kh * kw) + h * kw + w_c]).collect()
            })
        ];
        let w_scales: Vec<f32> = qw.scale.clone();
        let w_zps: Vec<i8> = qw.zero_point.clone();
        let n_quants = w_scales.len();
        let w_token = TokenBuffer4D::from(w_buf);

        let b_bias_tok = bias_constant_matrix(&bias_i32, bias_scale, qout, out_ch);
        let b_scale_tok = scale_constant_depthwise(&qin.scale[0], &qw.scale, qout);

        let weights_id = format_ident!("weights_{}", idx);
        let out_shape_ts = out_shape.iter().copied().collect::<Vec<_>>();
        let out_scales = &qout.scale;
        let out_zps = &qout.zero_point;
        quote! {
            const #weights_id: microflow::tensor::Tensor4D<i8, 1usize, #kh, #kw, #out_ch, #n_quants> =
                microflow::tensor::Tensor4D::new(#w_token, [#(#w_scales),*], [#(#w_zps),*]);
            let input: microflow::tensor::Tensor4D<_, #(#out_shape_ts),*, 1usize> =
                microflow::ops::depthwise_conv_2d(
                    input, &#weights_id,
                    [#(#out_scales),*], [#(#out_zps),*],
                    microflow::ops::DepthwiseConv2DOptions {
                        fused_activation: #fused_act,
                        view_padding: #padding_ts,
                        strides: (#stride_h, #stride_w),
                    },
                    (#b_bias_tok, #b_scale_tok)
                );
        }
    } else {
        // Regular conv: ONNX [out_ch, in_ch, kH, kW] → microflow [out_ch, kH, kW, in_ch]
        let w_buf: Vec<DMatrix<Vec<i8>>> = (0..out_ch).map(|oc| {
            DMatrix::from_fn(kh, kw, |h, w_c| {
                (0..in_ch_per_g).map(|ic| {
                    w_i8[oc * (in_ch_per_g * kh * kw) + ic * (kh * kw) + h * kw + w_c]
                }).collect()
            })
        }).collect();
        let w_scales: Vec<f32> = qw.scale.clone();
        let w_zps: Vec<i8> = qw.zero_point.clone();
        let n_quants = w_scales.len();
        let w_token = TokenBuffer4D::from(w_buf);

        let b_bias_tok = bias_constant_matrix(&bias_i32, bias_scale, qout, out_ch);
        let b_scale_tok = scale_constant_conv(&qin.scale[0], &qw.scale, qout);

        let filters_id = format_ident!("filters_{}", idx);
        let out_shape_ts = out_shape.iter().copied().collect::<Vec<_>>();
        let out_scales = &qout.scale;
        let out_zps = &qout.zero_point;
        quote! {
            const #filters_id: microflow::tensor::Tensor4D<i8, #out_ch, #kh, #kw, #in_ch_per_g, #n_quants> =
                microflow::tensor::Tensor4D::new(#w_token, [#(#w_scales),*], [#(#w_zps),*]);
            let input: microflow::tensor::Tensor4D<_, #(#out_shape_ts),*, 1usize> =
                microflow::ops::conv_2d(
                    input, &#filters_id,
                    [#(#out_scales),*], [#(#out_zps),*],
                    microflow::ops::Conv2DOptions {
                        fused_activation: #fused_act,
                        view_padding: #padding_ts,
                        strides: (#stride_h, #stride_w),
                    },
                    (#b_bias_tok, #b_scale_tok)
                );
        }
    }
}

fn gen_gemm(
    node: &OnnxNode,
    inits: &HashMap<String, OnnxTensor>,
    shape_map: &HashMap<String, Vec<usize>>,
    quant_map: &HashMap<String, Quant>,
    dq_src: &HashMap<String, String>,
    current_name: &str,
    out_name: &str,
    idx: usize,
    fused_act: &TokenStream2,
) -> TokenStream2 {
    let w_name = &node.input[1];
    let w_src = dq_src.get(w_name).map(|s| s.as_str()).unwrap_or(w_name);
    let w_init = inits.get(w_src)
        .unwrap_or_else(|| abort_call_site!("Gemm weight '{}' not found", w_src));
    let w_shape = tensor_shape(w_init);
    let (out_units, in_units) = (w_shape[0], w_shape[1]);

    let qw = quant_of(w_src, quant_map, dq_src);
    let qin = quant_of(current_name, quant_map, dq_src);
    let qout = quant_of(out_name, quant_map, dq_src);

    let w_i8: Vec<i8> = match w_init.data_type {
        DT_INT8 => i8_data(w_init),
        DT_FLOAT => {
            let floats = float_data(w_init);
            let (q, _) = quant_sym(&floats);
            q
        }
        dt => abort_call_site!("unsupported Gemm weight dtype {}", dt),
    };

    let out_shape = get_shape(out_name, shape_map);
    let (bias_i32, bias_scale) = read_bias(node, inits, dq_src, qin, qw, out_units);

    // Compute FullyConnected constants:
    // const0: bias_scale/out_scale * (bias - bias_zp)  [per output]
    // const1: in_scale * w_scale / out_scale            [scalar]
    // const2: sum-of-weights-per-row * in_zp            [per output]
    // const3: in_size * in_zp * w_zp                    [scalar]
    let in_zp = f32::from_subset(&qin.zero_point[0]);
    let w_zp = f32::from_subset(&qw.zero_point[0]);
    let c1: f32 = qin.scale[0] * qw.scale[0] / qout.scale[0];
    let c3: i32 = (in_units as f32 * in_zp * w_zp).round() as i32;

    let c0_vals: Vec<f32> = (0..out_units).map(|o| {
        let b = bias_i32.get(o).copied().unwrap_or(0);
        bias_scale / qout.scale[0] * b as f32
    }).collect();
    let c0_mat = DMatrix::from_column_slice(out_units, 1, &c0_vals);
    let c0_tok = TokenBuffer2D::from(c0_mat);

    // Row sums of weight matrix (each row = one output unit)
    let c2_vals: Vec<i32> = (0..out_units).map(|o| {
        let row_sum: i32 = (0..in_units).map(|i| w_i8[o * in_units + i] as i32).sum();
        (row_sum as f32 * in_zp).round() as i32
    }).collect();
    let c2_mat = DMatrix::from_column_slice(1, out_units, &c2_vals);
    let c2_tok = TokenBuffer2D::from(c2_mat);

    // Weight matrix: [out_units, in_units]
    let w_mat = DMatrix::from_row_slice(out_units, in_units, &w_i8);
    let w_tok = TokenBuffer2D::from(w_mat);
    let w_scale_v = &qw.scale;
    let w_zp_v = &qw.zero_point;
    let n_quants = w_scale_v.len();

    let weights_id = format_ident!("weights_{}", idx);
    let out_shape_ts = out_shape.iter().copied().collect::<Vec<_>>();
    let out_scales = &qout.scale;
    let out_zps = &qout.zero_point;
    let in_shape = get_shape(current_name, shape_map);
    let reshape = in_shape.len() != 2;
    let reshape_ts = if reshape { quote!(.into()) } else { quote!() };

    quote! {
        const #weights_id: microflow::tensor::Tensor2D<i8, #out_units, #in_units, #n_quants> =
            microflow::tensor::Tensor2D::new(#w_tok, [#(#w_scale_v),*], [#(#w_zp_v),*]);
        let input: microflow::tensor::Tensor2D<_, #(#out_shape_ts),*, 1usize> =
            microflow::ops::fully_connected(
                input #reshape_ts, &#weights_id,
                [#(#out_scales),*], [#(#out_zps),*],
                microflow::ops::FullyConnectedOptions {
                    fused_activation: #fused_act,
                },
                (#c0_tok, #c1, #c2_tok, #c3)
            );
    }
}

fn gen_avg_pool(
    node: &OnnxNode,
    shape_map: &HashMap<String, Vec<usize>>,
    qin: &Quant,
    qout: &Quant,
    current_name: &str,
    out_name: &str,
    fused_act: &TokenStream2,
) -> TokenStream2 {
    let out_shape = get_shape(out_name, shape_map);
    let in_shape = get_shape(current_name, shape_map);
    // GlobalAveragePool has no kernel_shape attribute; use the input spatial dims.
    let kh = attr_ints(&node.attribute, "kernel_shape")
        .map(|k| k[0] as usize)
        .unwrap_or_else(|| in_shape.get(1).copied().unwrap_or(1));
    let kw = attr_ints(&node.attribute, "kernel_shape")
        .map(|k| k[1] as usize)
        .unwrap_or_else(|| in_shape.get(2).copied().unwrap_or(1));
    let pads = attr_ints(&node.attribute, "pads").map(|p| p.to_vec()).unwrap_or_default();
    let padding_ts = if pads.iter().any(|&p| p != 0) {
        quote!(microflow::tensor::TensorViewPadding::Same)
    } else {
        quote!(microflow::tensor::TensorViewPadding::Valid)
    };
    let strides = attr_ints(&node.attribute, "strides")
        .map(|s| (s[0] as usize, s[1] as usize))
        .unwrap_or((1, 1));
    let (stride_h, stride_w) = strides;

    let c0 = qin.scale[0] / qout.scale[0];
    let in_zp = f32::from_subset(&qin.zero_point[0]);
    let c1 = f32::from_subset(&qout.zero_point[0]) - qin.scale[0] * in_zp / qout.scale[0];

    let out_shape_ts = out_shape.iter().copied().collect::<Vec<_>>();
    let out_scales = &qout.scale;
    let out_zps = &qout.zero_point;
    quote! {
        let input: microflow::tensor::Tensor4D<_, #(#out_shape_ts),*, 1usize> =
            microflow::ops::average_pool_2d(
                input,
                [#(#out_scales),*], [#(#out_zps),*],
                microflow::ops::AveragePool2DOptions {
                    fused_activation: #fused_act,
                    view_padding: #padding_ts,
                    strides: (#stride_h, #stride_w),
                    filter: (#kh, #kw),
                },
                (#c0, #c1)
            );
    }
}

fn gen_max_pool(
    node: &OnnxNode,
    shape_map: &HashMap<String, Vec<usize>>,
    qin: &Quant,
    qout: &Quant,
    out_name: &str,
    fused_act: &TokenStream2,
) -> TokenStream2 {
    let out_shape = get_shape(out_name, shape_map);
    let kh = attr_ints(&node.attribute, "kernel_shape").map(|k| k[0] as usize).unwrap_or(2);
    let kw = attr_ints(&node.attribute, "kernel_shape").map(|k| k[1] as usize).unwrap_or(2);
    let pads = attr_ints(&node.attribute, "pads").map(|p| p.to_vec()).unwrap_or_default();
    let padding_ts = if pads.iter().any(|&p| p != 0) {
        quote!(microflow::tensor::TensorViewPadding::Same)
    } else {
        quote!(microflow::tensor::TensorViewPadding::Valid)
    };
    let strides = attr_ints(&node.attribute, "strides")
        .map(|s| (s[0] as usize, s[1] as usize))
        .unwrap_or((1, 1));
    let (stride_h, stride_w) = strides;

    let c0 = qin.scale[0] / qout.scale[0];
    let in_zp = f32::from_subset(&qin.zero_point[0]);
    let c1 = f32::from_subset(&qout.zero_point[0]) - qin.scale[0] * in_zp / qout.scale[0];

    let out_shape_ts = out_shape.iter().copied().collect::<Vec<_>>();
    let out_scales = &qout.scale;
    let out_zps = &qout.zero_point;
    quote! {
        let input: microflow::tensor::Tensor4D<_, #(#out_shape_ts),*, 1usize> =
            microflow::ops::max_pool_2d(
                input,
                [#(#out_scales),*], [#(#out_zps),*],
                microflow::ops::MaxPool2DOptions {
                    fused_activation: #fused_act,
                    view_padding: #padding_ts,
                    strides: (#stride_h, #stride_w),
                    filter: (#kh, #kw),
                },
                (#c0, #c1)
            );
    }
}

fn gen_softmax(qout: &Quant, out_name: &str, shape_map: &HashMap<String, Vec<usize>>) -> TokenStream2 {
    let out_shape = get_shape(out_name, shape_map);
    let mut s = out_shape.clone();
    if s.len() == 1 { s.insert(0, 1); }
    let out_scales = &qout.scale;
    let out_zps = &qout.zero_point;
    quote! {
        let input: microflow::tensor::Tensor2D<_, #(#s),*, 1usize> =
            microflow::ops::softmax(input, [#(#out_scales),*], [#(#out_zps),*]);
    }
}

fn gen_reshape(out_name: &str, shape_map: &HashMap<String, Vec<usize>>) -> TokenStream2 {
    let out_shape = get_shape(out_name, shape_map);
    let tensor_ty = match out_shape.len() {
        2 => quote!(Tensor2D),
        4 => quote!(Tensor4D),
        _ => abort_call_site!("reshape to unsupported rank {}", out_shape.len()),
    };
    quote! {
        let input: microflow::tensor::#tensor_ty<_, #(#out_shape),*, 1usize> =
            microflow::ops::reshape(input);
    }
}

fn gen_resize(
    node: &OnnxNode,
    shape_map: &HashMap<String, Vec<usize>>,
    qin: &Quant,
    qout: &Quant,
    out_name: &str,
) -> TokenStream2 {
    let out_shape = get_shape(out_name, shape_map);
    let mode = attr_s(&node.attribute, "mode").unwrap_or_else(|| "linear".to_string());
    let c0 = qin.scale[0] / qout.scale[0];
    let c1 = f32::from_subset(&qout.zero_point[0])
        - c0 * f32::from_subset(&qin.zero_point[0]);
    let out_scales = &qout.scale;
    let out_zps = &qout.zero_point;
    if mode.contains("nearest") {
        quote! {
            let input: microflow::tensor::Tensor4D<_, #(#out_shape),*, 1usize> =
                microflow::ops::resize_nearest_neighbor(input, [#(#out_scales),*], [#(#out_zps),*], (#c0, #c1));
        }
    } else {
        quote! {
            let input: microflow::tensor::Tensor4D<_, #(#out_shape),*, 1usize> =
                microflow::ops::resize_bilinear(input, [#(#out_scales),*], [#(#out_zps),*], (#c0, #c1));
        }
    }
}

fn gen_concat(
    node: &OnnxNode,
    quant_map: &HashMap<String, Quant>,
    dq_src: &HashMap<String, String>,
    shape_map: &HashMap<String, Vec<usize>>,
    current_name: &str,
    out_name: &str,
) -> TokenStream2 {
    if node.input.len() != 2 {
        abort_call_site!("only 2-input Concat is supported for ONNX (got {})", node.input.len());
    }
    let axis = attr_i(&node.attribute, "axis").unwrap_or(-1);
    if axis != 3 && axis != -1 {
        abort_call_site!("only channel-axis (axis=3/-1) Concat is supported (got {})", axis);
    }

    let idx0 = &node.input[0];
    let idx1 = &node.input[1];
    let current_is_first = idx0 == current_name;
    let saved_name = if current_is_first { idx1 } else { idx0 };
    let saved_var = name_to_ident(saved_name);

    let qout = quant_of(out_name, quant_map, dq_src);
    let qt1 = quant_of(idx0, quant_map, dq_src);
    let qt2 = quant_of(idx1, quant_map, dq_src);

    let out_zp = f32::from_subset(&qout.zero_point[0]);
    let f1 = qt1.scale[0] / qout.scale[0];
    let o1 = out_zp - f1 * f32::from_subset(&qt1.zero_point[0]);
    let f2 = qt2.scale[0] / qout.scale[0];
    let o2 = out_zp - f2 * f32::from_subset(&qt2.zero_point[0]);

    let out_shape = get_shape(out_name, shape_map);
    let out_scales = &qout.scale;
    let out_zps = &qout.zero_point;

    if current_is_first {
        quote! {
            let input: microflow::tensor::Tensor4D<_, #(#out_shape),*, 1usize> =
                microflow::ops::concat(input, &#saved_var, [#(#out_scales),*], [#(#out_zps),*],
                    ((#f1, #o1), (#f2, #o2)));
        }
    } else {
        quote! {
            let input: microflow::tensor::Tensor4D<_, #(#out_shape),*, 1usize> =
                microflow::ops::concat(#saved_var, &input, [#(#out_scales),*], [#(#out_zps),*],
                    ((#f1, #o1), (#f2, #o2)));
        }
    }
}

fn gen_channel_split(
    node: &OnnxNode,
    _quant_map: &HashMap<String, Quant>,
    shape_map: &HashMap<String, Vec<usize>>,
    _current_name: &str,
) -> TokenStream2 {
    let n = node.output.len();
    assert!(
        n >= 2 && n <= 4,
        "only 2-, 3-, and 4-way Split are supported for ONNX (got {} outputs)",
        n
    );
    let axis = attr_i(&node.attribute, "axis").unwrap_or(-1);
    if axis != 3 && axis != -1 {
        abort_call_site!("only channel-axis Split is supported (axis={})", axis);
    }

    let s0 = get_shape(&node.output[0], shape_map);
    let (b, r, c) = (s0[0], s0[1], s0[2]);

    let chan_sizes: Vec<usize> = node.output.iter().map(|o| get_shape(o, shape_map)[3]).collect();
    let total: usize = chan_sizes.iter().sum();

    let type_annotations: Vec<_> = chan_sizes.iter().map(|ch| {
        quote! { microflow::tensor::Tensor4D<_, #b, #r, #c, #ch, 1usize> }
    }).collect();

    let saved_vars: Vec<_> = node.output[1..].iter().map(|o| name_to_ident(o)).collect();
    let ch_tokens: Vec<_> = chan_sizes.iter().map(|ch| quote!(#ch)).collect();
    let fn_name = format_ident!("channel_split_{}", n);

    quote! {
        let (input, #(#saved_vars),*): (
            #(#type_annotations),*
        ) = microflow::ops::#fn_name::<_, #b, #r, #c, #total, #(#ch_tokens),*>(input);
    }
}

// ─── Constant computation helpers ─────────────────────────────────────────────

/// Bias-term constant matrix: bias_scale / out_scale * (bias_val - bias_zp)
fn bias_constant_matrix(
    bias_i32: &[i32],
    bias_scale: f32,
    qout: &Quant,
    n: usize,
) -> TokenStream2 {
    let vals: Vec<f32> = (0..n).map(|i| {
        bias_scale / qout.scale[0] * bias_i32.get(i).copied().unwrap_or(0) as f32
    }).collect();
    let mat = DMatrix::from_column_slice(n, 1, &vals);
    let tok = TokenBuffer2D::from(mat);
    tok.to_token_stream()
}

/// Scale-constant vector for a regular Conv2D (per output channel):
/// in_scale * w_scale[c] / out_scale
fn scale_constant_conv(in_scale: &f32, w_scales: &[f32], qout: &Quant) -> TokenStream2 {
    let vals: Vec<f32> = w_scales.iter().map(|ws| in_scale * ws / qout.scale[0]).collect();
    let mat = DMatrix::from_column_slice(vals.len(), 1, &vals);
    TokenBuffer2D::from(mat).to_token_stream()
}

/// Scale-constant vector for a DepthwiseConv2D (per channel):
fn scale_constant_depthwise(in_scale: &f32, w_scales: &[f32], qout: &Quant) -> TokenStream2 {
    scale_constant_conv(in_scale, w_scales, qout)
}

/// Read bias tensor (node.input[2]) and return (i32 values, bias_scale).
fn read_bias(
    node: &OnnxNode,
    inits: &HashMap<String, OnnxTensor>,
    dq_src: &HashMap<String, String>,
    qin: &Quant,
    qw: &Quant,
    n: usize,
) -> (Vec<i32>, f32) {
    if node.input.len() >= 3 && !node.input[2].is_empty() {
        let b_src = dq_src.get(&node.input[2]).map(|s| s.as_str()).unwrap_or(&node.input[2]);
        if let Some(bi) = inits.get(b_src) {
            let data = match bi.data_type {
                DT_INT32 => i32_data(bi),
                DT_FLOAT => {
                    let floats = float_data(bi);
                    // bias_scale = in_scale * w_scale (standard quantized bias)
                    let bs = qin.scale[0] * qw.scale[0];
                    floats.iter().map(|&f| (f / bs).round() as i32).collect()
                }
                _ => vec![0i32; n],
            };
            let bs = qin.scale[0] * qw.scale[0];
            return (data, bs);
        }
    }
    (vec![0i32; n], qin.scale[0] * qw.scale[0])
}

// ─── Fused activation helper ──────────────────────────────────────────────────

/// Returns a TokenStream2 for the FusedActivation to emit given an activation node.
/// Clip with min=0, max=6 maps to Relu6; any other Clip / Relu maps to Relu.
fn fused_activation_from_node(node: &OnnxNode) -> TokenStream2 {
    if node.op_type == "Clip" {
        let min_val = attr_f(&node.attribute, "min").unwrap_or(0.0);
        let max_val = attr_f(&node.attribute, "max").unwrap_or(f32::MAX);
        if (min_val - 0.0).abs() < 1e-5 && (max_val - 6.0).abs() < 1e-3 {
            return quote!(microflow::activation::FusedActivation::Relu6);
        }
    }
    quote!(microflow::activation::FusedActivation::Relu)
}

fn attr_f(attrs: &[OnnxAttr], name: &str) -> Option<f32> {
    attrs.iter().find(|a| a.name == name).map(|a| a.f)
}

fn gen_standalone_activation(
    node: &OnnxNode,
    qout: &Quant,
    out_name: &str,
    shape_map: &HashMap<String, Vec<usize>>,
) -> TokenStream2 {
    let out_shape = get_shape(out_name, shape_map);
    let out_scales = &qout.scale;
    let out_zps = &qout.zero_point;

    if node.op_type == "Clip" {
        let min_val = attr_f(&node.attribute, "min").unwrap_or(0.0);
        let max_val = attr_f(&node.attribute, "max").unwrap_or(f32::MAX);
        if (min_val - 0.0).abs() > 1e-5 || (max_val - 6.0).abs() > 1e-3 {
            abort_call_site!(
                "standalone Clip(min={}, max={}) is not supported; \
                 only Clip(0, 6) = ReLU6 is handled as a fused activation",
                min_val, max_val
            );
        }
        // Clip(0, 6) standalone: emit relu as a conservative lower bound.
        // In fully-quantized models the upper clamp is typically already
        // captured by the output quantization range.
    }
    quote! {
        let input: microflow::tensor::Tensor4D<_, #(#out_shape),*, 1usize> =
            microflow::ops::relu(input, [#(#out_scales),*], [#(#out_zps),*]);
    }
}

// ─── New operator code generators ────────────────────────────────────────────

fn gen_add(
    node: &OnnxNode,
    quant_map: &HashMap<String, Quant>,
    dq_src: &HashMap<String, String>,
    shape_map: &HashMap<String, Vec<usize>>,
    current_name: &str,
    out_name: &str,
    fused_act: &TokenStream2,
) -> TokenStream2 {
    if node.input.len() != 2 {
        abort_call_site!("only 2-input Add is supported (got {})", node.input.len());
    }
    let idx0 = &node.input[0];
    let idx1 = &node.input[1];
    let current_is_first = idx0 == current_name;
    let saved_name = if current_is_first { idx1 } else { idx0 };
    let saved_var = name_to_ident(saved_name);

    let qt0 = quant_of(idx0, quant_map, dq_src);
    let qt1 = quant_of(idx1, quant_map, dq_src);
    let qout = quant_of(out_name, quant_map, dq_src);

    let c0 = qt0.scale[0] / qout.scale[0];
    let c1 = qt1.scale[0] / qout.scale[0];
    let c2 = f32::from_subset(&qout.zero_point[0])
        - c0 * f32::from_subset(&qt0.zero_point[0])
        - c1 * f32::from_subset(&qt1.zero_point[0]);

    let out_shape = get_shape(out_name, shape_map);
    let out_scales = &qout.scale;
    let out_zps = &qout.zero_point;

    if current_is_first {
        quote! {
            let input: microflow::tensor::Tensor4D<_, #(#out_shape),*, 1usize> =
                microflow::ops::add(input, &#saved_var,
                    [#(#out_scales),*], [#(#out_zps),*],
                    microflow::ops::AddOptions { fused_activation: #fused_act },
                    (#c0, #c1, #c2));
        }
    } else {
        quote! {
            let input: microflow::tensor::Tensor4D<_, #(#out_shape),*, 1usize> =
                microflow::ops::add(#saved_var, &input,
                    [#(#out_scales),*], [#(#out_zps),*],
                    microflow::ops::AddOptions { fused_activation: #fused_act },
                    (#c0, #c1, #c2));
        }
    }
}

fn gen_mul(
    node: &OnnxNode,
    quant_map: &HashMap<String, Quant>,
    dq_src: &HashMap<String, String>,
    shape_map: &HashMap<String, Vec<usize>>,
    current_name: &str,
    out_name: &str,
    fused_act: &TokenStream2,
) -> TokenStream2 {
    if node.input.len() != 2 {
        abort_call_site!("only 2-input Mul is supported (got {})", node.input.len());
    }
    let idx0 = &node.input[0];
    let idx1 = &node.input[1];
    let current_is_first = idx0 == current_name;
    let saved_name = if current_is_first { idx1 } else { idx0 };
    let saved_var = name_to_ident(saved_name);

    let qt0 = quant_of(idx0, quant_map, dq_src);
    let qt1 = quant_of(idx1, quant_map, dq_src);
    let qout = quant_of(out_name, quant_map, dq_src);

    let c0 = qt0.scale[0] * qt1.scale[0] / qout.scale[0];
    let zp0 = f32::from_subset(&qt0.zero_point[0]);
    let zp1 = f32::from_subset(&qt1.zero_point[0]);
    let d1 = c0 * zp0;
    let d2 = c0 * zp1;
    let c2 = f32::from_subset(&qout.zero_point[0]) + c0 * zp0 * zp1;

    let out_shape = get_shape(out_name, shape_map);
    let out_scales = &qout.scale;
    let out_zps = &qout.zero_point;

    if current_is_first {
        quote! {
            let input: microflow::tensor::Tensor4D<_, #(#out_shape),*, 1usize> =
                microflow::ops::mul(input, &#saved_var,
                    [#(#out_scales),*], [#(#out_zps),*],
                    microflow::ops::MulOptions { fused_activation: #fused_act },
                    (#c0, #d1, #d2, #c2));
        }
    } else {
        quote! {
            let input: microflow::tensor::Tensor4D<_, #(#out_shape),*, 1usize> =
                microflow::ops::mul(#saved_var, &input,
                    [#(#out_scales),*], [#(#out_zps),*],
                    microflow::ops::MulOptions { fused_activation: #fused_act },
                    (#c0, #d1, #d2, #c2));
        }
    }
}

fn gen_sigmoid(qout: &Quant, out_name: &str, shape_map: &HashMap<String, Vec<usize>>) -> TokenStream2 {
    let out_shape = get_shape(out_name, shape_map);
    let out_scales = &qout.scale;
    let out_zps = &qout.zero_point;
    quote! {
        let input: microflow::tensor::Tensor4D<_, #(#out_shape),*, 1usize> =
            microflow::ops::sigmoid(input, [#(#out_scales),*], [#(#out_zps),*]);
    }
}

fn gen_tanh(qout: &Quant, out_name: &str, shape_map: &HashMap<String, Vec<usize>>) -> TokenStream2 {
    let out_shape = get_shape(out_name, shape_map);
    let out_scales = &qout.scale;
    let out_zps = &qout.zero_point;
    quote! {
        let input: microflow::tensor::Tensor4D<_, #(#out_shape),*, 1usize> =
            microflow::ops::tanh(input, [#(#out_scales),*], [#(#out_zps),*]);
    }
}

fn gen_hard_swish(qout: &Quant, out_name: &str, shape_map: &HashMap<String, Vec<usize>>) -> TokenStream2 {
    let out_shape = get_shape(out_name, shape_map);
    let out_scales = &qout.scale;
    let out_zps = &qout.zero_point;
    quote! {
        let input: microflow::tensor::Tensor4D<_, #(#out_shape),*, 1usize> =
            microflow::ops::hard_swish(input, [#(#out_scales),*], [#(#out_zps),*]);
    }
}

fn gen_leaky_relu(
    node: &OnnxNode,
    qout: &Quant,
    out_name: &str,
    shape_map: &HashMap<String, Vec<usize>>,
) -> TokenStream2 {
    let out_shape = get_shape(out_name, shape_map);
    let out_scales = &qout.scale;
    let out_zps = &qout.zero_point;
    let alpha = attr_f(&node.attribute, "alpha").unwrap_or(0.01);
    quote! {
        let input: microflow::tensor::Tensor4D<_, #(#out_shape),*, 1usize> =
            microflow::ops::leaky_relu(input, #alpha, [#(#out_scales),*], [#(#out_zps),*]);
    }
}

fn gen_reduce_mean(
    node: &OnnxNode,
    shape_map: &HashMap<String, Vec<usize>>,
    qin: &Quant,
    qout: &Quant,
    out_name: &str,
) -> TokenStream2 {
    // axes attribute: expect [1,2] (H,W in NHWC) or [2,3] (H,W in NCHW)
    let axes = attr_ints(&node.attribute, "axes").map(|a| a.to_vec()).unwrap_or_default();
    assert!(
        axes == vec![1, 2] || axes == vec![2, 3],
        "only H+W ReduceMean (axes=[1,2] or [2,3]) is supported, got {:?}",
        axes
    );

    let out_shape = get_shape(out_name, shape_map);
    // Infer spatial extent from output shape (shape should be [B,1,1,C] or similar)
    // Use the axes to determine which dimensions were reduced.
    // The in_shape rows/cols are recovered from the axes applied to out_shape context.
    // We can't recover them from out_shape alone, so look up current_name's shape via out_shape.
    // For simplicity, use constant factor from qin/qout and let R*C be inferred
    // by noting that out_shape is [B,1,1,C] so the spatial reduction is total.
    // We look for the reduced dimensions in the input via the shape_map.
    //
    // Since we don't have the current input name here, use a conservative estimate:
    // c0 = in_scale / out_scale  (without the 1/R*C factor — caller must provide full c0)
    // Actually we need R*C. The runtime function signature takes (c0, c1) where
    // c0 = in_scale / (out_scale * R * C). We derive R*C from the shape_map if available,
    // otherwise default to 1 (GlobalAveragePool-style where R*C is encoded in out_shape context).
    //
    // Best effort: look for an intermediate shape in shape_map that matches out_name prefix.
    // For now compute without R*C and rely on shape_map lookup by examining node inputs.
    let in_name = node.input.first().cloned().unwrap_or_default();
    let in_shape = shape_map.get(&in_name).cloned().unwrap_or_else(|| out_shape.clone());
    let (rows, cols) = if axes == vec![1, 2] {
        (in_shape.get(1).copied().unwrap_or(1), in_shape.get(2).copied().unwrap_or(1))
    } else {
        (in_shape.get(2).copied().unwrap_or(1), in_shape.get(3).copied().unwrap_or(1))
    };
    let c0 = qin.scale[0] / (qout.scale[0] * (rows * cols) as f32);
    let c1 = f32::from_subset(&qout.zero_point[0]);

    let out_scales = &qout.scale;
    let out_zps = &qout.zero_point;
    quote! {
        let input: microflow::tensor::Tensor4D<_, #(#out_shape),*, 1usize> =
            microflow::ops::reduce_mean(
                input,
                [#(#out_scales),*], [#(#out_zps),*],
                (#c0, #c1),
            );
    }
}

fn gen_conv_transpose(
    node: &OnnxNode,
    inits: &HashMap<String, OnnxTensor>,
    shape_map: &HashMap<String, Vec<usize>>,
    quant_map: &HashMap<String, Quant>,
    dq_src: &HashMap<String, String>,
    current_name: &str,
    out_name: &str,
    idx: usize,
    fused_act: &TokenStream2,
) -> TokenStream2 {
    let strides = attr_ints(&node.attribute, "strides")
        .map(|s| (s[0] as usize, s[1] as usize))
        .unwrap_or((1, 1));
    let (stride_h, stride_w) = strides;

    // ONNX ConvTranspose weights: [in_ch, out_ch/groups, kH, kW]
    let w_name = &node.input[1];
    let w_src = dq_src.get(w_name).map(|s| s.as_str()).unwrap_or(w_name);
    let w_init = inits.get(w_src)
        .unwrap_or_else(|| abort_call_site!("ConvTranspose weight '{}' not found", w_src));
    let w_shape = tensor_shape(w_init); // [in_ch, out_ch, kH, kW]
    let (in_ch, out_ch, kh, kw) = (w_shape[0], w_shape[1], w_shape[2], w_shape[3]);

    let qw = quant_of(w_src, quant_map, dq_src);
    let qin = quant_of(current_name, quant_map, dq_src);
    let qout = quant_of(out_name, quant_map, dq_src);

    let w_i8: Vec<i8> = match w_init.data_type {
        DT_INT8 => i8_data(w_init),
        DT_FLOAT => {
            let floats = float_data(w_init);
            let (q, _) = quant_sym(&floats);
            q
        }
        dt => abort_call_site!("unsupported ConvTranspose weight dtype {}", dt),
    };

    // Reorder ONNX [in_ch, out_ch, kH, kW] → microflow [out_ch, kH, kW, in_ch]
    let w_buf: Vec<DMatrix<Vec<i8>>> = (0..out_ch).map(|oc| {
        DMatrix::from_fn(kh, kw, |h, wc| {
            (0..in_ch).map(|ic| {
                w_i8[ic * (out_ch * kh * kw) + oc * (kh * kw) + h * kw + wc]
            }).collect()
        })
    }).collect();

    let w_scales: Vec<f32> = qw.scale.clone();
    let w_zps: Vec<i8> = qw.zero_point.clone();
    let n_quants = w_scales.len();
    let w_token = TokenBuffer4D::from(w_buf);

    let (bias_i32, bias_scale) = read_bias(node, inits, dq_src, qin, qw, out_ch);
    let b_bias_tok = bias_constant_matrix(&bias_i32, bias_scale, qout, out_ch);
    let b_scale_tok = scale_constant_conv(&qin.scale[0], &qw.scale, qout);

    let out_shape = get_shape(out_name, shape_map);
    let out_shape_ts = out_shape.iter().copied().collect::<Vec<_>>();
    let out_scales = &qout.scale;
    let out_zps = &qout.zero_point;

    let filters_id = format_ident!("filters_{}", idx);
    quote! {
        const #filters_id: microflow::tensor::Tensor4D<i8, #out_ch, #kh, #kw, #in_ch, #n_quants> =
            microflow::tensor::Tensor4D::new(#w_token, [#(#w_scales),*], [#(#w_zps),*]);
        let input: microflow::tensor::Tensor4D<_, #(#out_shape_ts),*, 1usize> =
            microflow::ops::transpose_conv_2d(
                input, &#filters_id,
                [#(#out_scales),*], [#(#out_zps),*],
                microflow::ops::TransposeConv2DOptions {
                    fused_activation: #fused_act,
                    strides: (#stride_h, #stride_w),
                },
                (#b_bias_tok, #b_scale_tok)
            );
    }
}
