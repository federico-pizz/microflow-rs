use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote, ToTokens};
use simba::scalar::SupersetOf;

use crate::quantize::TokenQuantized;
use crate::tensor::TokenTensor4D;
use crate::tflite_flatbuffers::tflite::{Operator, Tensor};
use flatbuffers::{ForwardsUOffset, Vector};

/// Represents the tokenized version of the `Concat` operator (axis=3, channel concatenation).
pub(crate) struct TokenConcat<T: TokenQuantized> {
    pub(crate) output: TokenTensor4D<T>,
    /// Index of the non-current (saved) input tensor, used to form the variable name.
    pub(crate) saved_tensor_idx: usize,
    /// If true, the current `input` is concatenated first (input1), the saved tensor second.
    /// If false, the saved tensor is concatenated first (input1), the current `input` second.
    pub(crate) current_is_first: bool,
    /// Pre-computed requantization constants: ((factor1, offset1), (factor2, offset2)).
    pub(crate) constants: ((f32, f32), (f32, f32)),
}

/// Parses the [`TokenConcat`] struct from the given operator.
///
/// # Arguments
/// * `operator` - The model operator as an [`Operator`]
/// * `tensors` - The model tensors as a [`Vector<ForwardsUOffset<Tensor>>`]
/// * `current_tensor_idx` - The tensor index currently held in the `input` variable
///
pub(crate) fn parse<T: TokenQuantized>(
    operator: Operator,
    tensors: Vector<ForwardsUOffset<Tensor>>,
    current_tensor_idx: usize,
) -> TokenConcat<T> {
    TokenConcat::<T>::new(operator, tensors, current_tensor_idx)
}

impl<T: TokenQuantized> TokenConcat<T> {
    pub(crate) fn new(
        operator: Operator,
        tensors: Vector<ForwardsUOffset<Tensor>>,
        current_tensor_idx: usize,
    ) -> Self {
        let inputs = operator.inputs().unwrap();
        assert_eq!(
            inputs.len(),
            2,
            "only 2-input CONCATENATION is supported, got {} inputs",
            inputs.len()
        );

        let idx0 = inputs.get(0) as usize;
        let idx1 = inputs.get(1) as usize;

        // Determine which input is the current sequential flow and which is saved.
        let (current_is_first, saved_tensor_idx) = if idx0 == current_tensor_idx {
            (true, idx1)
        } else if idx1 == current_tensor_idx {
            (false, idx0)
        } else {
            panic!(
                "CONCATENATION: neither input tensor ({}, {}) matches the current tensor ({}). \
                 Only skip-connection patterns are supported.",
                idx0, idx1, current_tensor_idx
            );
        };

        let options = operator
            .builtin_options_as_concatenation_options()
            .unwrap();
        let axis = options.axis();
        assert!(
            axis == 3 || axis == -1,
            "only axis=3 or axis=-1 (channel) concatenation is supported, got axis={}",
            axis
        );

        // input1 is whichever comes first in concat order (may differ from the current flow).
        let t1 = TokenTensor4D::<T>::from_empty_tensor(
            tensors.get(if current_is_first { idx0 } else { idx0 }),
        );
        let t2 = TokenTensor4D::<T>::from_empty_tensor(
            tensors.get(if current_is_first { idx1 } else { idx1 }),
        );
        let output = TokenTensor4D::<T>::from_empty_tensor(
            tensors.get(operator.outputs().unwrap().get(0) as usize),
        );

        let out_zp = f32::from_subset(&output.zero_point[0]);

        let factor1 = t1.scale[0] / output.scale[0];
        let offset1 = out_zp - factor1 * f32::from_subset(&t1.zero_point[0]);
        let factor2 = t2.scale[0] / output.scale[0];
        let offset2 = out_zp - factor2 * f32::from_subset(&t2.zero_point[0]);

        Self {
            output,
            saved_tensor_idx,
            current_is_first,
            constants: ((factor1, offset1), (factor2, offset2)),
        }
    }
}

impl<T: TokenQuantized> ToTokens for TokenConcat<T> {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let saved_ident = format_ident!("tensor_{}", self.saved_tensor_idx);
        let output_shape = &self.output.shape;
        let output_scale = self.output.scale[0];
        let output_zero_point = self.output.zero_point[0];
        let ((f1, o1), (f2, o2)) = self.constants;

        let ts = if self.current_is_first {
            quote! {
                let input: microflow::tensor::Tensor4D<_, #(#output_shape),*, 1usize> =
                    microflow::ops::concat(
                        input,
                        &#saved_ident,
                        [#output_scale],
                        [#output_zero_point],
                        ((#f1, #o1), (#f2, #o2)),
                    );
            }
        } else {
            // saved tensor goes first (its channels appear before the current input's channels)
            quote! {
                let input: microflow::tensor::Tensor4D<_, #(#output_shape),*, 1usize> =
                    microflow::ops::concat(
                        #saved_ident,
                        &input,
                        [#output_scale],
                        [#output_zero_point],
                        ((#f1, #o1), (#f2, #o2)),
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

    fn setup() -> TokenConcat<i8> {
        TokenConcat {
            output: TokenTensor4D {
                buffer: TokenBuffer4D::new(),
                shape: vec![1, 2, 2, 3],
                scale: vec![0.1],
                zero_point: vec![0i8],
            },
            saved_tensor_idx: 5,
            current_is_first: true,
            constants: ((1.0, 0.0), (2.0, 0.0)),
        }
    }

    #[test]
    fn concat_to_tokens_current_first() {
        let layer = setup();
        assert_eq!(
            layer.to_token_stream().to_string(),
            quote! {
                let input: microflow::tensor::Tensor4D<_, 1usize, 2usize, 2usize, 3usize, 1usize> =
                    microflow::ops::concat(
                        input,
                        &tensor_5,
                        [0.1f32],
                        [0i8],
                        ((1f32, 0f32), (2f32, 0f32)),

                    );
            }
            .to_string()
        );
    }

    #[test]
    fn concat_to_tokens_saved_first() {
        let mut layer = setup();
        layer.current_is_first = false;
        assert_eq!(
            layer.to_token_stream().to_string(),
            quote! {
                let input: microflow::tensor::Tensor4D<_, 1usize, 2usize, 2usize, 3usize, 1usize> =
                    microflow::ops::concat(
                        tensor_5,
                        &input,
                        [0.1f32],
                        [0i8],
                        ((1f32, 0f32), (2f32, 0f32)),

                    );
            }
            .to_string()
        );
    }
}
