use flatbuffers::{ForwardsUOffset, Vector};
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote, ToTokens};

use crate::quantize::TokenQuantized;
use crate::tensor::TokenTensor4D;
use crate::tflite_flatbuffers::tflite::{Operator, Tensor, TensorType};

/// Represents the tokenized version of a standalone `RELU` or `RELU6` operator.
pub(crate) struct TokenRelu<T: TokenQuantized> {
    pub(crate) output: TokenTensor4D<T>,
    /// `true` for ReLU6, `false` for plain ReLU.
    pub(crate) is_relu6: bool,
}

pub(crate) fn parse_relu(
    operator: Operator,
    tensors: Vector<ForwardsUOffset<Tensor>>,
) -> Box<dyn ToTokens> {
    parse_impl(operator, tensors, false)
}

pub(crate) fn parse_relu6(
    operator: Operator,
    tensors: Vector<ForwardsUOffset<Tensor>>,
) -> Box<dyn ToTokens> {
    parse_impl(operator, tensors, true)
}

fn parse_impl<'a>(
    operator: Operator<'a>,
    tensors: Vector<'a, ForwardsUOffset<Tensor<'a>>>,
    is_relu6: bool,
) -> Box<dyn ToTokens> {
    let inputs = operator.inputs().unwrap();
    let input_type = tensors.get(inputs.get(0) as usize).type_();
    match input_type {
        TensorType::INT8 => Box::new(TokenRelu::<i8>::new(operator, tensors, is_relu6)),
        TensorType::UINT8 => Box::new(TokenRelu::<u8>::new(operator, tensors, is_relu6)),
        _ => unimplemented!(),
    }
}

impl<T: TokenQuantized> TokenRelu<T> {
    pub(crate) fn new(
        operator: Operator,
        tensors: Vector<ForwardsUOffset<Tensor>>,
        is_relu6: bool,
    ) -> Self {
        let output = TokenTensor4D::<T>::from_empty_tensor(
            tensors.get(operator.outputs().unwrap().get(0) as usize),
        );
        Self { output, is_relu6 }
    }
}

impl<T: TokenQuantized> ToTokens for TokenRelu<T> {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let output_shape = &self.output.shape;
        let output_scale = &self.output.scale;
        let output_zero_point = &self.output.zero_point;
        let fn_name = if self.is_relu6 {
            format_ident!("relu6")
        } else {
            format_ident!("relu")
        };

        let ts = quote! {
            let input: microflow::tensor::Tensor4D<_, #(#output_shape),*, 1usize> =
                microflow::ops::#fn_name(input, [#(#output_scale),*], [#(#output_zero_point),*]);
        };
        ts.to_tokens(tokens);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::TokenBuffer4D;

    fn setup(is_relu6: bool) -> TokenRelu<i8> {
        TokenRelu {
            output: TokenTensor4D {
                buffer: TokenBuffer4D::new(),
                shape: vec![1, 4, 4, 8],
                scale: vec![0.5],
                zero_point: vec![0i8],
            },
            is_relu6,
        }
    }

    #[test]
    fn relu_to_tokens() {
        let layer = setup(false);
        assert_eq!(
            layer.to_token_stream().to_string(),
            quote! {
                let input: microflow::tensor::Tensor4D<_, 1usize, 4usize, 4usize, 8usize, 1usize> =
                    microflow::ops::relu(input, [0.5f32], [0i8]);
            }
            .to_string()
        );
    }

    #[test]
    fn relu6_to_tokens() {
        let layer = setup(true);
        assert_eq!(
            layer.to_token_stream().to_string(),
            quote! {
                let input: microflow::tensor::Tensor4D<_, 1usize, 4usize, 4usize, 8usize, 1usize> =
                    microflow::ops::relu6(input, [0.5f32], [0i8]);
            }
            .to_string()
        );
    }
}
