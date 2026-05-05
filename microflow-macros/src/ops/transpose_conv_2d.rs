use flatbuffers::{ForwardsUOffset, Vector};
use nalgebra::DMatrix;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote, ToTokens};

use crate::activation::TokenFusedActivation;
use crate::buffer::TokenBuffer2D;
use crate::quantize::TokenQuantized;
use crate::tensor::{TokenTensor2D, TokenTensor4D};
use crate::tflite_flatbuffers::tflite::{Buffer, Operator, Tensor, TensorType};

/// Represents the tokenized version of the `TransposeConv2D` operator.
pub(crate) struct TokenTransposeConv2D<T: TokenQuantized> {
    pub(crate) filters: TokenTensor4D<T>,
    pub(crate) output: TokenTensor4D<T>,
    pub(crate) fused_activation: TokenFusedActivation,
    pub(crate) strides: (usize, usize),
    pub(crate) constants: (TokenBuffer2D<f32>, TokenBuffer2D<f32>),
    pub(crate) index: usize,
}

pub(crate) fn parse(
    operator: Operator,
    tensors: Vector<ForwardsUOffset<Tensor>>,
    buffers: Vector<ForwardsUOffset<Buffer>>,
    index: usize,
) -> Box<dyn ToTokens> {
    let inputs = operator.inputs().unwrap();
    // TFLite TRANSPOSE_CONV: inputs[2] is the data tensor.
    let input_type = tensors.get(inputs.get(2) as usize).type_();
    match input_type {
        TensorType::INT8 => Box::new(TokenTransposeConv2D::<i8>::new(
            operator, tensors, buffers, index,
        )),
        TensorType::UINT8 => Box::new(TokenTransposeConv2D::<u8>::new(
            operator, tensors, buffers, index,
        )),
        _ => unimplemented!(),
    }
}

impl<T: TokenQuantized> TokenTransposeConv2D<T> {
    pub(crate) fn new(
        operator: Operator,
        tensors: Vector<ForwardsUOffset<Tensor>>,
        buffers: Vector<ForwardsUOffset<Buffer>>,
        index: usize,
    ) -> Self {
        let inputs = operator.inputs().unwrap();
        // TFLite TRANSPOSE_CONV input layout:
        //   inputs[0] = output_shape (int32 constant, ignored — read from output tensor)
        //   inputs[1] = filter weights
        //   inputs[2] = data input
        //   inputs[3] = bias (optional, may be absent or -1)
        let input = TokenTensor4D::<T>::from_empty_tensor(tensors.get(inputs.get(2) as usize));
        let filters =
            TokenTensor4D::from_buffered_tensor(tensors.get(inputs.get(1) as usize), buffers);
        let output = TokenTensor4D::<T>::from_empty_tensor(
            tensors.get(operator.outputs().unwrap().get(0) as usize),
        );
        let options = operator.builtin_options_as_transpose_conv_options().unwrap();

        let biases = if inputs.len() >= 4 {
            let bias_idx = inputs.get(3) as usize;
            // A value of -1 means the bias tensor is absent.
            if bias_idx != usize::MAX {
                Some(TokenTensor2D::from_buffered_tensor(
                    tensors.get(bias_idx),
                    buffers,
                ))
            } else {
                None
            }
        } else {
            None
        };

        let constants = Self::preprocess(&input, &filters, biases.as_ref(), &output);
        Self {
            filters,
            output,
            fused_activation: options.fused_activation_function().into(),
            strides: (options.stride_h() as usize, options.stride_w() as usize),
            constants,
            index,
        }
    }

    fn preprocess(
        input: &TokenTensor4D<T>,
        filters: &TokenTensor4D<T>,
        biases: Option<&TokenTensor2D<i32>>,
        output: &TokenTensor4D<T>,
    ) -> (TokenBuffer2D<f32>, TokenBuffer2D<f32>) {
        let out_ch = filters.shape[0];
        (
            TokenBuffer2D::from(DMatrix::from_fn(out_ch, 1, |b, _| {
                if let Some(bias) = biases {
                    let bs = bias.scale.get(b).copied().unwrap_or(bias.scale[0]);
                    let bzp = bias.zero_point.get(b).copied().unwrap_or(bias.zero_point[0]);
                    bs / output.scale[0] * (bias.buffer[b] - bzp) as f32
                } else {
                    0.0f32
                }
            })),
            TokenBuffer2D::from(DMatrix::from_fn(filters.scale.len(), 1, |b, _| {
                input.scale[0] * filters.scale[b] / output.scale[0]
            })),
        )
    }
}

impl<T: TokenQuantized> ToTokens for TokenTransposeConv2D<T> {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let filters_ident = format_ident!("filters_{}", self.index);
        let filters_type = self.filters.type_tokens();
        let filters = &self.filters;
        let output_shape = &self.output.shape;
        let output_scale = &self.output.scale;
        let output_zero_point = &self.output.zero_point;
        let fused_activation = self.fused_activation;
        let (strides_0, strides_1) = self.strides;
        let (constants_0, constants_1) = &self.constants;

        let ts = quote! {
            const #filters_ident: #filters_type = #filters;
            let input: microflow::tensor::Tensor4D<_, #(#output_shape),*, 1usize> =
                microflow::ops::transpose_conv_2d(
                    input,
                    &#filters_ident,
                    [#(#output_scale),*],
                    [#(#output_zero_point),*],
                    microflow::ops::TransposeConv2DOptions {
                        fused_activation: #fused_activation,
                        strides: (#strides_0, #strides_1),
                    },
                    (#constants_0, #constants_1)
                );
        };
        ts.to_tokens(tokens);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{TokenBuffer2D, TokenBuffer4D};
    use nalgebra::dmatrix;

    fn setup() -> TokenTransposeConv2D<i8> {
        TokenTransposeConv2D {
            filters: TokenTensor4D {
                buffer: TokenBuffer4D::from(vec![
                    dmatrix![vec![1i8, 2], vec![3, 4]],
                    dmatrix![vec![5i8, 6], vec![7, 8]],
                ]),
                shape: vec![2, 2, 1, 2],
                scale: vec![0.1, 0.1],
                zero_point: vec![0i8, 0],
            },
            output: TokenTensor4D {
                buffer: TokenBuffer4D::new(),
                shape: vec![1, 3, 2, 2],
                scale: vec![0.01],
                zero_point: vec![0i8],
            },
            fused_activation: TokenFusedActivation::None,
            strides: (2, 1),
            constants: (
                TokenBuffer2D::from(dmatrix![0.0f32; 0.0]),
                TokenBuffer2D::from(dmatrix![1.0f32; 1.0]),
            ),
            index: 0,
        }
    }

    #[test]
    fn transpose_conv_2d_to_tokens() {
        let layer = setup();
        let filters = &layer.filters;
        let fused_activation = layer.fused_activation;
        let (constants_0, constants_1) = &layer.constants;
        assert_eq!(
            layer.to_token_stream().to_string(),
            quote! {
                const filters_0: microflow::tensor::Tensor4D<i8, 2usize, 2usize, 1usize, 2usize, 2usize> = #filters;
                let input: microflow::tensor::Tensor4D<_, 1usize, 3usize, 2usize, 2usize, 1usize> =
                    microflow::ops::transpose_conv_2d(
                        input,
                        &filters_0,
                        [0.01f32],
                        [0i8],
                        microflow::ops::TransposeConv2DOptions {
                            fused_activation: #fused_activation,
                            strides: (2usize, 1usize),
                        },
                        (#constants_0, #constants_1)
                    );
            }
            .to_string()
        );
    }
}
