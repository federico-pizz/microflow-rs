use flatbuffers::{ForwardsUOffset, Vector};
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote, ToTokens};
use simba::scalar::SupersetOf;

use crate::activation::TokenFusedActivation;
use crate::quantize::TokenQuantized;
use crate::tensor::TokenTensor4D;
use crate::tflite_flatbuffers::tflite::{Operator, Tensor};

/// Represents the tokenized version of the `Add` (element-wise addition) operator.
pub(crate) struct TokenAdd<T: TokenQuantized> {
    pub(crate) output: TokenTensor4D<T>,
    /// Index of the non-current (saved) input tensor.
    pub(crate) saved_tensor_idx: usize,
    /// True when inputs()[0] is the current sequential `input`.
    pub(crate) current_is_first: bool,
    pub(crate) fused_activation: TokenFusedActivation,
    /// (c0, c1, c2) where c0 = s0/s_out, c1 = s1/s_out,
    /// c2 = zp_out - c0*zp0 - c1*zp1  (indices match TFLite input order).
    pub(crate) constants: (f32, f32, f32),
}

pub(crate) fn parse<T: TokenQuantized>(
    operator: Operator,
    tensors: Vector<ForwardsUOffset<Tensor>>,
    current_tensor_idx: usize,
) -> TokenAdd<T> {
    TokenAdd::<T>::new(operator, tensors, current_tensor_idx)
}

impl<T: TokenQuantized> TokenAdd<T> {
    pub(crate) fn new(
        operator: Operator,
        tensors: Vector<ForwardsUOffset<Tensor>>,
        current_tensor_idx: usize,
    ) -> Self {
        let inputs = operator.inputs().unwrap();
        let idx0 = inputs.get(0) as usize;
        let idx1 = inputs.get(1) as usize;

        let current_is_first = idx0 == current_tensor_idx;
        let saved_tensor_idx = if current_is_first { idx1 } else { idx0 };

        let t0 = TokenTensor4D::<T>::from_empty_tensor(tensors.get(idx0));
        let t1 = TokenTensor4D::<T>::from_empty_tensor(tensors.get(idx1));
        let output = TokenTensor4D::<T>::from_empty_tensor(
            tensors.get(operator.outputs().unwrap().get(0) as usize),
        );
        let options = operator.builtin_options_as_add_options().unwrap();
        let constants = Self::preprocess(&t0, &t1, &output);
        Self {
            output,
            saved_tensor_idx,
            current_is_first,
            fused_activation: options.fused_activation_function().into(),
            constants,
        }
    }

    fn preprocess(
        t0: &TokenTensor4D<T>,
        t1: &TokenTensor4D<T>,
        output: &TokenTensor4D<T>,
    ) -> (f32, f32, f32) {
        let c0 = t0.scale[0] / output.scale[0];
        let c1 = t1.scale[0] / output.scale[0];
        let c2 = f32::from_subset(&output.zero_point[0])
            - c0 * f32::from_subset(&t0.zero_point[0])
            - c1 * f32::from_subset(&t1.zero_point[0]);
        (c0, c1, c2)
    }
}

impl<T: TokenQuantized> ToTokens for TokenAdd<T> {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let saved_ident = format_ident!("tensor_{}", self.saved_tensor_idx);
        let output_shape = &self.output.shape;
        let output_scale = &self.output.scale;
        let output_zero_point = &self.output.zero_point;
        let fused_activation = self.fused_activation;
        let (c0, c1, c2) = self.constants;

        // Always pass in TFLite input order so c0/c1 align with the right tensor.
        let ts = if self.current_is_first {
            quote! {
                let input: microflow::tensor::Tensor4D<_, #(#output_shape),*, 1usize> =
                    microflow::ops::add(
                        input, &#saved_ident,
                        [#(#output_scale),*], [#(#output_zero_point),*],
                        microflow::ops::AddOptions { fused_activation: #fused_activation },
                        (#c0, #c1, #c2),
                    );
            }
        } else {
            quote! {
                let input: microflow::tensor::Tensor4D<_, #(#output_shape),*, 1usize> =
                    microflow::ops::add(
                        #saved_ident, &input,
                        [#(#output_scale),*], [#(#output_zero_point),*],
                        microflow::ops::AddOptions { fused_activation: #fused_activation },
                        (#c0, #c1, #c2),
                    );
            }
        };
        ts.to_tokens(tokens);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::TokenBuffer4D;

    fn setup() -> TokenAdd<i8> {
        TokenAdd {
            output: TokenTensor4D {
                buffer: TokenBuffer4D::new(),
                shape: vec![1, 2, 2, 4],
                scale: vec![0.1],
                zero_point: vec![0i8],
            },
            saved_tensor_idx: 5,
            current_is_first: true,
            fused_activation: TokenFusedActivation::None,
            constants: (1.0, 2.0, 0.0),
        }
    }

    #[test]
    fn add_to_tokens_current_first() {
        let layer = setup();
        let fused_activation = layer.fused_activation;
        assert_eq!(
            layer.to_token_stream().to_string(),
            quote! {
                let input: microflow::tensor::Tensor4D<_, 1usize, 2usize, 2usize, 4usize, 1usize> =
                    microflow::ops::add(
                        input, &tensor_5,
                        [0.1f32], [0i8],
                        microflow::ops::AddOptions { fused_activation: #fused_activation },
                        (1f32, 2f32, 0f32),
                    );
            }
            .to_string()
        );
    }

    #[test]
    fn add_to_tokens_saved_first() {
        let mut layer = setup();
        layer.current_is_first = false;
        let fused_activation = layer.fused_activation;
        assert_eq!(
            layer.to_token_stream().to_string(),
            quote! {
                let input: microflow::tensor::Tensor4D<_, 1usize, 2usize, 2usize, 4usize, 1usize> =
                    microflow::ops::add(
                        tensor_5, &input,
                        [0.1f32], [0i8],
                        microflow::ops::AddOptions { fused_activation: #fused_activation },
                        (1f32, 2f32, 0f32),
                    );
            }
            .to_string()
        );
    }
}
