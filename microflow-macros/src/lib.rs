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
                #layers
                input
            }
        }
    };

    fs::write("target/microflow-expansion.rs", ts.to_string()).ok();

    ts.into()
}
