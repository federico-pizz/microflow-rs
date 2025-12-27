use crate::quantize::TokenQuantized;
use crate::tensor::TokenTensor4D;
use crate::tflite_flatbuffers::tflite::{Operator, Tensor, TensorType};
use flatbuffers::{ForwardsUOffset, Vector};
use proc_macro2::TokenStream as TokenStream2;
use quote::{quote, ToTokens};

/// Represents the tokenized version of the `ReLU` operator.
pub(crate) struct TokenRelu<T: TokenQuantized> {
    pub(crate) output: TokenTensor4D<T>,
}

/// Parses the [`TokenRelu`] struct from the given operator.
///
/// # Arguments
/// * `operator` - The model operator as an [`Operator`]
/// * `tensors` - The model tensors as a [`Vector<ForwardsUOffset<Tensor>>`]
///
pub(crate) fn parse(
    operator: Operator,
    tensors: Vector<ForwardsUOffset<Tensor>>,
) -> Box<dyn ToTokens> {
    let inputs = operator.inputs().unwrap();
    let input_type = tensors.get(inputs.get(0) as usize).type_();
    match input_type {
        TensorType::INT8 => Box::new(TokenRelu::<i8>::new(operator, tensors)),
        TensorType::UINT8 => Box::new(TokenRelu::<u8>::new(operator, tensors)),
        _ => unimplemented!(),
    }
}

impl<T: TokenQuantized> TokenRelu<T> {
    /// Builds the [`TokenRelu`] operator from the given model operator and tensors.
    ///
    /// # Arguments
    /// * `operator` - The model operator as an [`Operator`]
    /// * `tensors` - The model tensors as a [`Vector<ForwardsUOffset<Tensor>>`]
    ///
    pub(crate) fn new(operator: Operator, tensors: Vector<ForwardsUOffset<Tensor>>) -> Self {
        let output = TokenTensor4D::from_empty_tensor(
            tensors.get(operator.outputs().unwrap().get(0) as usize),
        );
        Self { output }
    }
}

impl<T: TokenQuantized> ToTokens for TokenRelu<T> {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let output_scale = &self.output.scale;
        let output_zero_point = &self.output.zero_point;

        let ts = quote! {
            microflow::ops::relu_in_place(&mut input, [#(#output_scale),*], [#(#output_zero_point),*]);
        };
        ts.to_tokens(tokens);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::TokenBuffer4D;

    fn setup() -> TokenRelu<i8> {
        TokenRelu {
            output: TokenTensor4D {
                buffer: TokenBuffer4D::new(),
                shape: vec![1, 2, 2, 1],
                scale: vec![0.5],
                zero_point: vec![0],
            },
        }
    }

    #[test]
    fn relu_to_tokens() {
        let layer = setup();
        assert_eq!(
            layer.to_token_stream().to_string(),
            quote! {
                microflow::ops::relu_in_place(&mut input, [0.5f32], [0i8]);
            }
            .to_string()
        )
    }
}