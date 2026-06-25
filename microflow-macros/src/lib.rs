//! [![crates.io](https://img.shields.io/crates/v/microflow-macros)](https://crates.io/crates/microflow-macros)
//! [![docs.rs](https://img.shields.io/docsrs/microflow-macros)](https://docs.rs/microflow-macros)
//! [![github](https://img.shields.io/github/actions/workflow/status/matteocarnelos/microflow-rs/cargo.yml?branch=main)](https://github.com/matteocarnelos/microflow-rs/actions/workflows/cargo.yml)
//!
//! Macro crate of the [MicroFlow](https://github.com/matteocarnelos/microflow-rs) inference engine, namely, the MicroFlow compiler.

extern crate proc_macro;

use proc_macro::TokenStream;
use proc_macro_error::{abort_call_site, proc_macro_error};
use std::collections::HashSet;
use std::fs;

use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote, ToTokens};
use syn::{parse_macro_input, ItemStruct};

use crate::tflite_flatbuffers::tflite::TensorType;
use ops::buffer_reuse::BufferReuse;
use ops::*;
use structmeta::StructMeta;
use syn::LitStr;
use tflite_flatbuffers::tflite::{root_as_model, BuiltinOperator};

mod activation;
mod buffer;
mod onnx;
mod ops;
mod quantize;
mod tensor;
#[path = "../flatbuffers/tflite_generated.rs"]
#[allow(unused_imports)]
#[allow(clippy::all)]
mod tflite_flatbuffers;

#[derive(StructMeta)]
struct Args {
    #[struct_meta(unnamed)]
    path: LitStr,
}

/// The entry point of MicroFlow.
/// This attribute-like procedural macro can be placed on `structs` to implement the `predict()`
/// function based on the given model.
/// The macro takes as input the path of the model, which can be in TensorFlow Lite format
/// (`.tflite`) or ONNX format (`.onnx`).
#[proc_macro_error]
#[proc_macro_attribute]
pub fn model(args: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(args as Args);
    let item = parse_macro_input!(item as ItemStruct);

    let path = args.path.value();
    let buf = fs::read(&path).unwrap_or_else(|_| {
        abort_call_site!(
            "couldn't find '{}', please provide a valid path",
            &path
        )
    });

    if path.ends_with(".onnx") {
        return model_from_onnx(&buf, &item);
    }

    model_from_tflite(&buf, &item)
}

fn model_from_onnx(buf: &[u8], item: &ItemStruct) -> TokenStream {
    let parsed = onnx::parse_onnx(buf);
    let ident = &item.ident;

    let input_shape = &parsed.input_shape;
    let input_quant = &parsed.input_quant;
    let output_shape = &parsed.output_shape;

    let input_scales = &input_quant.scale;
    let input_zps = &input_quant.zero_point;

    let input_tensor = if input_shape.len() == 2 { quote!(Tensor2D) } else { quote!(Tensor4D) };
    let input_buffer = if input_shape.len() == 2 { quote!(Buffer2D) } else { quote!(Buffer4D) };
    let output_tensor = if output_shape.len() == 2 { quote!(Tensor2D) } else { quote!(Tensor4D) };
    let output_buffer = if output_shape.len() == 2 { quote!(Buffer2D) } else { quote!(Buffer4D) };

    let mut layers = TokenStream2::new();
    for ts in &parsed.layers {
        ts.to_tokens(&mut layers);
    }

    let ts = quote! {
        #item
        impl #ident {
            pub fn predict(input: microflow::buffer::#input_buffer<f32, #(#input_shape),*>) -> microflow::buffer::#output_buffer<f32, #(#output_shape),*> {
                let input = microflow::tensor::#input_tensor::quantize(input, [#(#input_scales),*], [#(#input_zps),*]);
                Self::predict_inner(input).dequantize()
            }

            pub fn predict_quantized(input: microflow::buffer::#input_buffer<i8, #(#input_shape),*>) -> microflow::buffer::#output_buffer<f32, #(#output_shape),*> {
                let input = microflow::tensor::#input_tensor::new(input, [#(#input_scales),*], [#(#input_zps),*]);
                Self::predict_inner(input).dequantize()
            }

            fn predict_inner(input: microflow::tensor::#input_tensor<i8, #(#input_shape),*, 1usize>) -> microflow::tensor::#output_tensor<i8, #(#output_shape),*, 1usize> {
                #layers
                input
            }
        }
    };

    fs::write("target/microflow-expansion.rs", ts.to_string()).ok();
    ts.into()
}

fn model_from_tflite(buf: &[u8], item: &ItemStruct) -> TokenStream {
    let model = root_as_model(buf).unwrap_or_else(|_| {
        abort_call_site!("invalid model, please provide a valid TensorFlow Lite model")
    });

    let ident = &item.ident;

    let subgraph = model.subgraphs().unwrap().get(0);
    let tensors = subgraph.tensors().unwrap();
    let buffers = model.buffers().unwrap();

    let input = tensors.get(subgraph.inputs().unwrap().get(0) as usize);
    let mut input_shape: Vec<_> = input.shape().unwrap().iter().map(|e| e as usize).collect();
    if input_shape.len() == 1 {
        input_shape.insert(0, 1);
    }
    let input_type = match input.type_() {
        TensorType::INT8 => quote!(i8),
        TensorType::UINT8 => quote!(u8),
        _ => unimplemented!(),
    };
    let input_tensor = match input_shape.len() {
        2 => quote!(Tensor2D),
        4 => quote!(Tensor4D),
        _ => unimplemented!(),
    };
    let input_buffer = match input_shape.len() {
        2 => quote!(Buffer2D),
        4 => quote!(Buffer4D),
        _ => unimplemented!(),
    };
    let input_scale: Vec<_> = input
        .quantization()
        .unwrap()
        .scale()
        .unwrap()
        .iter()
        .map(|e| e.to_token_stream())
        .collect();
    let input_zero_point: Vec<_> = match input.type_() {
        TensorType::INT8 => input
            .quantization()
            .unwrap()
            .zero_point()
            .unwrap()
            .iter()
            .map(|e| (e as i8).to_token_stream())
            .collect(),
        TensorType::UINT8 => input
            .quantization()
            .unwrap()
            .zero_point()
            .unwrap()
            .iter()
            .map(|e| (e as u8).to_token_stream())
            .collect(),
        _ => unimplemented!(),
    };

    let operators = subgraph.operators().unwrap();

    // Phase 1: find tensor indices that must be saved (cloned) because they are consumed by a
    // CONCATENATION op, and identify channel-split non-primary outputs (already named, no clone needed).
    let mut channel_split_secondary_outputs: HashSet<usize> = HashSet::new();
    for operator in operators.iter() {
        let code = BuiltinOperator(
            model
                .operator_codes()
                .unwrap()
                .get(operator.opcode_index() as usize)
                .deprecated_builtin_code() as i32,
        );
        if code == BuiltinOperator::SPLIT {
            let outputs = operator.outputs().unwrap();
            // Primary output is outputs[0]; all others are saved by the destructuring, not cloned.
            for k in 1..outputs.len() {
                channel_split_secondary_outputs.insert(outputs.get(k) as usize);
            }
        }
    }

    let mut tensors_to_save: HashSet<usize> = HashSet::new();
    for operator in operators.iter() {
        let code = BuiltinOperator(
            model
                .operator_codes()
                .unwrap()
                .get(operator.opcode_index() as usize)
                .deprecated_builtin_code() as i32,
        );
        if code == BuiltinOperator::CONCATENATION
            || code == BuiltinOperator::ADD
            || code == BuiltinOperator::MUL
        {
            let inputs = operator.inputs().unwrap();
            for k in 0..inputs.len() {
                let idx = inputs.get(k) as usize;
                // Skip tensors already bound by a channel-split destructuring.
                if !channel_split_secondary_outputs.contains(&idx) {
                    tensors_to_save.insert(idx);
                }
            }
        }
    }

    // Phase 2: iterate operators, track the current tensor index, and emit code.
    let input_idx = subgraph.inputs().unwrap().get(0) as usize;
    let mut current_tensor_idx = input_idx;
    let mut layers = TokenStream2::new();

    for (index, operator) in operators.iter().enumerate() {
        let output_idx = operator.outputs().unwrap().get(0) as usize;

        let op_code = BuiltinOperator(
            model
                .operator_codes()
                .unwrap()
                .get(operator.opcode_index() as usize)
                .deprecated_builtin_code() as i32,
        );

        // TFLite SPLIT maps to channel_split: it produces multiple outputs and uses destructuring.
        if op_code == BuiltinOperator::SPLIT {
            let input_type = tensors
                .get(operator.inputs().unwrap().get(1) as usize)
                .type_();
            let layer: Box<dyn ToTokens> = match input_type {
                TensorType::INT8 => Box::new(channel_split::parse::<i8>(operator, tensors, buffers)),
                TensorType::UINT8 => Box::new(channel_split::parse::<u8>(operator, tensors, buffers)),
                _ => abort_call_site!("unsupported tensor type for SPLIT"),
            };
            layer.to_tokens(&mut layers);
            // Primary output is outputs[0]; update current_tensor_idx.
            current_tensor_idx = output_idx;
            // If the primary output will also be needed by a future CONCAT, save it.
            if tensors_to_save.contains(&output_idx) {
                let ident = format_ident!("tensor_{}", output_idx);
                quote! { let #ident = input.clone(); }.to_tokens(&mut layers);
            }
            continue;
        }

        let layer: Box<dyn ToTokens> = match op_code {
            BuiltinOperator::FULLY_CONNECTED => {
                fully_connected::parse(operator, tensors, buffers, index)
            }
            BuiltinOperator::DEPTHWISE_CONV_2D => {
                depthwise_conv_2d::parse(operator, tensors, buffers, index)
            }
            BuiltinOperator::CONV_2D => conv_2d::parse(operator, tensors, buffers, index),
            BuiltinOperator::AVERAGE_POOL_2D => average_pool_2d::parse(operator, tensors),
            BuiltinOperator::MAX_POOL_2D => max_pool_2d::parse(operator, tensors),
            BuiltinOperator::SOFTMAX => softmax::parse(operator, tensors),
            BuiltinOperator::RESHAPE => Box::new(reshape::parse(operator, tensors)),
            BuiltinOperator::RESIZE_BILINEAR => {
                resize_bilinear::parse(operator, tensors, buffers, index)
            }
            BuiltinOperator::RESIZE_NEAREST_NEIGHBOR => {
                resize_nearest_neighbor::parse(operator, tensors, buffers, index)
            }
            BuiltinOperator::CONCATENATION => {
                let input_type = tensors
                    .get(operator.inputs().unwrap().get(0) as usize)
                    .type_();
                match input_type {
                    TensorType::INT8 => Box::new(concat::parse::<i8>(
                        operator,
                        tensors,
                        current_tensor_idx,
                    )),
                    TensorType::UINT8 => Box::new(concat::parse::<u8>(
                        operator,
                        tensors,
                        current_tensor_idx,
                    )),
                    _ => abort_call_site!("unsupported tensor type for CONCATENATION"),
                }
            }
            BuiltinOperator::ADD => {
                let input_type = tensors
                    .get(operator.inputs().unwrap().get(0) as usize)
                    .type_();
                match input_type {
                    TensorType::INT8 => Box::new(add::parse::<i8>(
                        operator,
                        tensors,
                        current_tensor_idx,
                    )),
                    TensorType::UINT8 => Box::new(add::parse::<u8>(
                        operator,
                        tensors,
                        current_tensor_idx,
                    )),
                    _ => abort_call_site!("unsupported tensor type for ADD"),
                }
            }
            BuiltinOperator::MUL => {
                let input_type = tensors
                    .get(operator.inputs().unwrap().get(0) as usize)
                    .type_();
                match input_type {
                    TensorType::INT8 => Box::new(mul::parse::<i8>(
                        operator,
                        tensors,
                        current_tensor_idx,
                    )),
                    TensorType::UINT8 => Box::new(mul::parse::<u8>(
                        operator,
                        tensors,
                        current_tensor_idx,
                    )),
                    _ => abort_call_site!("unsupported tensor type for MUL"),
                }
            }
            BuiltinOperator::RELU => relu::parse_relu(operator, tensors),
            BuiltinOperator::RELU6 => relu::parse_relu6(operator, tensors),
            BuiltinOperator::LOGISTIC => sigmoid::parse(operator, tensors),
            BuiltinOperator::TANH => tanh::parse(operator, tensors),
            BuiltinOperator::HARD_SWISH => hard_swish::parse(operator, tensors),
            BuiltinOperator::LEAKY_RELU => leaky_relu::parse(operator, tensors, buffers),
            BuiltinOperator::MEAN => reduce_mean::parse(operator, tensors, buffers),
            BuiltinOperator::TRANSPOSE_CONV => {
                transpose_conv_2d::parse(operator, tensors, buffers, index)
            }
            unsupported_op => abort_call_site!("unsupported operator: {:?}", unsupported_op),
        };
        layer.to_tokens(&mut layers);

        // Advance the sequential tensor tracker to this op's output.
        current_tensor_idx = output_idx;

        // If this output will be consumed by a later CONCATENATION as a non-current tensor,
        // emit a clone so it remains accessible after the next op moves `input`.
        if tensors_to_save.contains(&output_idx) {
            let ident = format_ident!("tensor_{}", output_idx);
            quote! { let #ident = input.clone(); }.to_tokens(&mut layers);
        }
    }

    // Buffer-reuse (ping-pong) codegen. When the `buffer-reuse` feature is on
    // and the model is an eligible linear chain of in-place-capable Tensor4D ops
    // (no saved/cloned tensors, no channel splits), build a `predict_inner` body
    // that ping-pongs between two persistent enum buffers instead of shadowing a
    // distinct-typed local per layer. This collapses peak stack from the sum of
    // all intermediate tensors toward ~2×MAX. See ops/buffer_reuse.rs.
    let buffer_reuse_body: Option<TokenStream2> = if cfg!(feature = "buffer-reuse")
        && tensors_to_save.is_empty()
        && channel_split_secondary_outputs.is_empty()
        && input_shape.len() == 4
        && operators.len() >= 2
    {
        // Collect an in-place step for every operator; bail out (None) if any op
        // is not Tensor4D->Tensor4D in-place-capable, falling back to the shadow
        // chain emitted above.
        let mut steps: Vec<Box<dyn BufferReuse>> = Vec::new();
        let mut eligible = true;
        for (index, operator) in operators.iter().enumerate() {
            let op_code = BuiltinOperator(
                model
                    .operator_codes()
                    .unwrap()
                    .get(operator.opcode_index() as usize)
                    .deprecated_builtin_code() as i32,
            );
            let input_t = tensors
                .get(operator.inputs().unwrap().get(0) as usize)
                .type_();
            let boxed: Option<Box<dyn BufferReuse>> = match op_code {
                BuiltinOperator::CONV_2D => match input_t {
                    TensorType::INT8 => Some(Box::new(conv_2d::TokenConv2D::<i8>::new(
                        operator, tensors, buffers, index,
                    ))),
                    TensorType::UINT8 => Some(Box::new(conv_2d::TokenConv2D::<u8>::new(
                        operator, tensors, buffers, index,
                    ))),
                    _ => None,
                },
                BuiltinOperator::DEPTHWISE_CONV_2D => match input_t {
                    TensorType::INT8 => Some(Box::new(
                        depthwise_conv_2d::TokenDepthwiseConv2D::<i8>::new(
                            operator, tensors, buffers, index,
                        ),
                    )),
                    TensorType::UINT8 => Some(Box::new(
                        depthwise_conv_2d::TokenDepthwiseConv2D::<u8>::new(
                            operator, tensors, buffers, index,
                        ),
                    )),
                    _ => None,
                },
                BuiltinOperator::MAX_POOL_2D => match input_t {
                    TensorType::INT8 => {
                        Some(Box::new(max_pool_2d::TokenMaxPool2D::<i8>::new(operator, tensors)))
                    }
                    TensorType::UINT8 => {
                        Some(Box::new(max_pool_2d::TokenMaxPool2D::<u8>::new(operator, tensors)))
                    }
                    _ => None,
                },
                BuiltinOperator::RESIZE_NEAREST_NEIGHBOR => match input_t {
                    TensorType::INT8 => Some(Box::new(
                        resize_nearest_neighbor::TokenResizeNearestNeighbor::<i8>::new(
                            operator, tensors, buffers, index,
                        ),
                    )),
                    TensorType::UINT8 => Some(Box::new(
                        resize_nearest_neighbor::TokenResizeNearestNeighbor::<u8>::new(
                            operator, tensors, buffers, index,
                        ),
                    )),
                    _ => None,
                },
                _ => None,
            };
            match boxed {
                Some(b) => steps.push(b),
                None => {
                    eligible = false;
                    break;
                }
            }
        }

        if eligible {
            Some(build_pingpong_body(&steps, &input_shape, &input_type))
        } else {
            None
        }
    } else {
        None
    };

    let output = tensors.get(subgraph.outputs().unwrap().get(0) as usize);
    let mut output_shape: Vec<_> = output.shape().unwrap().iter().map(|e| e as usize).collect();
    if output_shape.len() == 1 {
        output_shape.insert(0, 1);
    }
    let output_type = match output.type_() {
        TensorType::INT8 => quote!(i8),
        TensorType::UINT8 => quote!(u8),
        _ => unimplemented!(),
    };
    let output_tensor = match output_shape.len() {
        2 => quote!(Tensor2D),
        4 => quote!(Tensor4D),
        _ => unimplemented!(),
    };
    let output_buffer = match output_shape.len() {
        2 => quote!(Buffer2D),
        4 => quote!(Buffer4D),
        _ => unimplemented!(),
    };

    let predict_inner_body = match buffer_reuse_body {
        Some(body) => body,
        None => quote! {
            #layers
            input
        },
    };

    let ts = quote! {
        #item
        impl #ident {
            pub fn predict(input: microflow::buffer::#input_buffer<f32, #(#input_shape),*>) -> microflow::buffer::#output_buffer<f32, #(#output_shape),*> {
                let input = microflow::tensor::#input_tensor::quantize(input, [#(#input_scale),*], [#(#input_zero_point),*]);
                Self::predict_inner(input).dequantize()
            }

            pub fn predict_quantized(input: microflow::buffer::#input_buffer<#input_type, #(#input_shape),*>) -> microflow::buffer::#output_buffer<f32, #(#output_shape),*> {
                let input = microflow::tensor::#input_tensor::new(input, [#(#input_scale),*], [#(#input_zero_point),*]);
                Self::predict_inner(input).dequantize()
            }

            fn predict_inner(input: microflow::tensor::#input_tensor<#input_type, #(#input_shape),*, 1usize>) -> microflow::tensor::#output_tensor<#output_type, #(#output_shape),*, 1usize> {
                #predict_inner_body
            }
        }
    };

    fs::write("target/microflow-expansion.rs", ts.to_string()).ok();

    ts.into()
}

/// Builds the ping-pong `predict_inner` body for an eligible linear chain.
///
/// Emits a per-model `enum Buf` (one variant per distinct intermediate tensor
/// shape), the hoisted op constants, one `#[inline(always)] fn step_k` per
/// layer (each fills a fresh `out` of the destination shape from the source via
/// the op's `*_into` variant, then re-tags the destination buffer), and a driver
/// that ping-pongs between two persistent `Buf` locals. See ops/buffer_reuse.rs
/// for why the fixed-size enum + in-place fill collapses peak stack to ~2×MAX.
fn build_pingpong_body(
    steps: &[Box<dyn BufferReuse>],
    input_shape: &[usize],
    input_type: &TokenStream2,
) -> TokenStream2 {
    // shapes[0] = model input, shapes[k] = output of operator k (1-indexed).
    let mut shapes: Vec<Vec<usize>> = vec![input_shape.to_vec()];
    for step in steps {
        shapes.push(step.br_out_shape());
    }

    // Deduplicate shapes into enum variants; `shape_variant[i]` is the variant
    // index of `shapes[i]`.
    let mut unique: Vec<Vec<usize>> = Vec::new();
    let mut shape_variant: Vec<usize> = Vec::new();
    for sh in &shapes {
        let idx = unique.iter().position(|u| u == sh).unwrap_or_else(|| {
            unique.push(sh.clone());
            unique.len() - 1
        });
        shape_variant.push(idx);
    }

    // enum Buf { V0(Tensor4D<..>), V1(..), .. }
    let variants = unique.iter().enumerate().map(|(i, sh)| {
        let variant = format_ident!("V{}", i);
        let (r, c, ch) = (sh[1], sh[2], sh[3]);
        quote! { #variant(microflow::tensor::Tensor4D<#input_type, 1usize, #r, #c, #ch, 1usize>) }
    });
    let enum_def = quote! { enum Buf { #(#variants),* } };

    let src_ident = format_ident!("src_t");
    let out_ident = format_ident!("out");

    // Hoisted consts (filters/weights) and one step fn per layer.
    let mut consts = TokenStream2::new();
    let mut step_fns = TokenStream2::new();
    for (k, step) in steps.iter().enumerate() {
        let step_no = k + 1;
        let src_variant = format_ident!("V{}", shape_variant[k]);
        let dst_variant = format_ident!("V{}", shape_variant[k + 1]);
        let dst_shape = &shapes[k + 1];
        let (dr, dc, dch) = (dst_shape[1], dst_shape[2], dst_shape[3]);
        let step_fn = format_ident!("step_{}", step_no);
        let zero_ident = format_ident!("zero_buf_{}", step_no);
        let call = step.br_call(&src_ident, &out_ident);
        consts.extend(step.br_consts());
        // A compile-time `Buf` already tagged to the destination variant and
        // holding an all-zero tensor, stored in rodata. Re-tagging `dst` by
        // copying this const (rather than building a zero tensor and an enum
        // wrapper on the stack) avoids any large stack temporary per step.
        consts.extend(quote! {
            const #zero_ident: Buf = Buf::#dst_variant(
                microflow::tensor::Tensor4D::new(
                    [nalgebra::Matrix::from_array_storage(nalgebra::ArrayStorage(
                        [[[0 as #input_type; #dch]; #dr]; #dc],
                    ))],
                    [0f32],
                    [0 as #input_type],
                ),
            );
        });
        // Re-tag `dst` from the const, then fill that payload in place via the
        // op's `*_into` variant. Keeping the only large storage inside the two
        // persistent buffers (no moved-out local, no stack-built zero or enum
        // wrapper) is what lets peak stack collapse toward ~2×MAX rather than
        // the sum.
        step_fns.extend(quote! {
            #[inline(never)]
            fn #step_fn(dst: &mut Buf, src: &Buf) {
                let #src_ident = match src {
                    Buf::#src_variant(t) => t,
                    _ => ::core::unreachable!(),
                };
                *dst = #zero_ident;
                let #out_ident = match dst {
                    Buf::#dst_variant(t) => t,
                    _ => ::core::unreachable!(),
                };
                #call
            }
        });
    }

    // Driver: two persistent buffers, alternating destination each step.
    let n = steps.len();
    let in_variant = format_ident!("V{}", shape_variant[0]);
    let out_variant = format_ident!("V{}", shape_variant[n]);
    let (ir, ic, ich) = (input_shape[1], input_shape[2], input_shape[3]);

    let mut calls = TokenStream2::new();
    for step_no in 1..=n {
        let step_fn = format_ident!("step_{}", step_no);
        if step_no % 2 == 1 {
            calls.extend(quote! { #step_fn(&mut buf_b, &buf_a); });
        } else {
            calls.extend(quote! { #step_fn(&mut buf_a, &buf_b); });
        }
    }
    let final_buf = if n % 2 == 1 {
        quote!(buf_b)
    } else {
        quote!(buf_a)
    };

    quote! {
        #enum_def
        #consts
        #step_fns
        // `buf_b`'s initial value is overwritten by the first step; build it from
        // a rodata const so the placeholder costs no stack.
        const ZERO_BUF_IN: Buf = Buf::#in_variant(
            microflow::tensor::Tensor4D::new(
                [nalgebra::Matrix::from_array_storage(nalgebra::ArrayStorage(
                    [[[0 as #input_type; #ich]; #ir]; #ic],
                ))],
                [0f32],
                [0 as #input_type],
            ),
        );
        let mut buf_a = Buf::#in_variant(input);
        let mut buf_b = ZERO_BUF_IN;
        #calls
        let input = match #final_buf {
            Buf::#out_variant(t) => t,
            _ => ::core::unreachable!(),
        };
        input
    }
}
