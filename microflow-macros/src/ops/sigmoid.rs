use flatbuffers::{ForwardsUOffset, Vector};
use proc_macro2::TokenStream as TokenStream2;
use quote::{quote, ToTokens};

use crate::quantize::TokenQuantized;
use crate::tensor::TokenTensor4D;
use crate::tflite_flatbuffers::tflite::{Operator, Tensor, TensorType};

/// Represents the tokenized version of the `Sigmoid` (LOGISTIC) operator.
pub(crate) struct TokenSigmoid<T: TokenQuantized> {
    pub(crate) output: TokenTensor4D<T>,
}

pub(crate) fn parse(
    operator: Operator,
    tensors: Vector<ForwardsUOffset<Tensor>>,
) -> Box<dyn ToTokens> {
    let inputs = operator.inputs().unwrap();
    let input_type = tensors.get(inputs.get(0) as usize).type_();
    match input_type {
        TensorType::INT8 => Box::new(TokenSigmoid::<i8>::new(operator, tensors)),
        TensorType::UINT8 => Box::new(TokenSigmoid::<u8>::new(operator, tensors)),
        _ => unimplemented!(),
    }
}

impl<T: TokenQuantized> TokenSigmoid<T> {
    pub(crate) fn new(
        operator: Operator,
        tensors: Vector<ForwardsUOffset<Tensor>>,
    ) -> Self {
        let output = TokenTensor4D::<T>::from_empty_tensor(
            tensors.get(operator.outputs().unwrap().get(0) as usize),
        );
        Self { output }
    }
}

impl<T: TokenQuantized> ToTokens for TokenSigmoid<T> {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let output_shape = &self.output.shape;
        let output_scale = &self.output.scale;
        let output_zero_point = &self.output.zero_point;

        let ts = quote! {
            let input: microflow::tensor::Tensor4D<_, #(#output_shape),*, 1usize> =
                microflow::ops::sigmoid(input, [#(#output_scale),*], [#(#output_zero_point),*]);
        };
        ts.to_tokens(tokens);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::TokenBuffer4D;

    fn setup() -> TokenSigmoid<i8> {
        TokenSigmoid {
            output: TokenTensor4D {
                buffer: TokenBuffer4D::new(),
                shape: vec![1, 4, 4, 8],
                scale: vec![1.0 / 256.0],
                zero_point: vec![-128i8],
            },
        }
    }

    #[test]
    fn sigmoid_to_tokens() {
        let layer = setup();
        assert_eq!(
            layer.to_token_stream().to_string(),
            quote! {
                let input: microflow::tensor::Tensor4D<_, 1usize, 4usize, 4usize, 8usize, 1usize> =
                    microflow::ops::sigmoid(input, [0.00390625f32], [-128i8]);
            }
            .to_string()
        );
    }
}
