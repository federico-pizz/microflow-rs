use std::mem::size_of;

use flatbuffers::{ForwardsUOffset, Vector};
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote, ToTokens};

use crate::quantize::TokenQuantized;
use crate::tensor::TokenTensor4D;
use crate::tflite_flatbuffers::tflite::{Buffer, Operator, Tensor};

/// Represents the tokenized version of an N-way channel split operator (N = 2, 3, or 4).
pub(crate) struct TokenChannelSplit<T: TokenQuantized> {
    /// Each output tensor paired with its tensor index (used to form `tensor_N` variable names).
    /// outputs[0] becomes the new `input`; outputs[1..] are saved as named variables.
    pub(crate) outputs: Vec<(TokenTensor4D<T>, usize)>,
}

/// Parses a [`TokenChannelSplit`] from the given TFLite SPLIT operator.
pub(crate) fn parse<T: TokenQuantized>(
    operator: Operator,
    tensors: Vector<ForwardsUOffset<Tensor>>,
    buffers: Vector<ForwardsUOffset<Buffer>>,
) -> TokenChannelSplit<T> {
    TokenChannelSplit::<T>::new(operator, tensors, buffers)
}

impl<T: TokenQuantized> TokenChannelSplit<T> {
    pub(crate) fn new(
        operator: Operator,
        tensors: Vector<ForwardsUOffset<Tensor>>,
        buffers: Vector<ForwardsUOffset<Buffer>>,
    ) -> Self {
        let inputs = operator.inputs().unwrap();
        let outputs = operator.outputs().unwrap();

        let options = operator.builtin_options_as_split_options().unwrap();
        let n = options.num_splits() as usize;
        assert!(
            n >= 2 && n <= 4,
            "only 2-, 3-, and 4-way SPLIT are supported, got {} splits",
            n
        );
        assert_eq!(outputs.len() as usize, n, "SPLIT outputs count mismatch");

        // TFLite SPLIT: inputs[0] = axis tensor, inputs[1] = data tensor
        let axis_tensor = tensors.get(inputs.get(0) as usize);
        let axis_bytes = buffers
            .get(axis_tensor.buffer() as usize)
            .data()
            .unwrap()
            .bytes();
        let axis = i32::from_le_bytes(axis_bytes[..size_of::<i32>()].try_into().unwrap());
        assert!(
            axis == 3 || axis == -1,
            "only channel-axis (axis=3 or axis=-1) SPLIT is supported, got axis={}",
            axis
        );

        let out_list: Vec<(TokenTensor4D<T>, usize)> = (0..n)
            .map(|k| {
                let idx = outputs.get(k) as usize;
                (TokenTensor4D::<T>::from_empty_tensor(tensors.get(idx)), idx)
            })
            .collect();

        Self { outputs: out_list }
    }
}

impl<T: TokenQuantized> ToTokens for TokenChannelSplit<T> {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let n = self.outputs.len();
        let (out0_tensor, _) = &self.outputs[0];
        let shape0 = &out0_tensor.shape;
        let (batches, rows, cols) = (shape0[0], shape0[1], shape0[2]);

        let chan_sizes: Vec<usize> = self.outputs.iter().map(|(t, _)| t.shape[3]).collect();
        let total_chans: usize = chan_sizes.iter().sum();

        // Build the turbofish const-generic list: <_, B, R, C, TOTAL, CH1, CH2, ...>
        let turbofish = {
            let ch_tokens: Vec<_> = chan_sizes.iter().map(|c| quote!(#c)).collect();
            quote! { <_, #batches, #rows, #cols, #total_chans, #(#ch_tokens),*> }
        };

        // Build the destructuring pattern: (input, tensor_N2, tensor_N3, ...)
        let secondary_idents: Vec<_> = self.outputs[1..]
            .iter()
            .map(|(_, idx)| format_ident!("tensor_{}", idx))
            .collect();

        // Build the type annotation for each output
        let type_annotations: Vec<_> = self
            .outputs
            .iter()
            .map(|(t, _)| {
                let ch = t.shape[3];
                quote! { microflow::tensor::Tensor4D<_, #batches, #rows, #cols, #ch, 1usize> }
            })
            .collect();

        let fn_name = format_ident!("channel_split_{}", n);

        let ts = quote! {
            let (input, #(#secondary_idents),*): (
                #(#type_annotations),*
            ) = microflow::ops::#fn_name::#turbofish(input);
        };
        ts.to_tokens(tokens);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::TokenBuffer4D;

    fn setup_2way() -> TokenChannelSplit<i8> {
        TokenChannelSplit {
            outputs: vec![
                (
                    TokenTensor4D {
                        buffer: TokenBuffer4D::new(),
                        shape: vec![1, 4, 4, 16],
                        scale: vec![0.1],
                        zero_point: vec![0i8],
                    },
                    6,
                ),
                (
                    TokenTensor4D {
                        buffer: TokenBuffer4D::new(),
                        shape: vec![1, 4, 4, 16],
                        scale: vec![0.1],
                        zero_point: vec![0i8],
                    },
                    7,
                ),
            ],
        }
    }

    fn setup_3way() -> TokenChannelSplit<i8> {
        TokenChannelSplit {
            outputs: vec![
                (
                    TokenTensor4D {
                        buffer: TokenBuffer4D::new(),
                        shape: vec![1, 4, 4, 8],
                        scale: vec![0.1],
                        zero_point: vec![0i8],
                    },
                    5,
                ),
                (
                    TokenTensor4D {
                        buffer: TokenBuffer4D::new(),
                        shape: vec![1, 4, 4, 8],
                        scale: vec![0.1],
                        zero_point: vec![0i8],
                    },
                    6,
                ),
                (
                    TokenTensor4D {
                        buffer: TokenBuffer4D::new(),
                        shape: vec![1, 4, 4, 8],
                        scale: vec![0.1],
                        zero_point: vec![0i8],
                    },
                    7,
                ),
            ],
        }
    }

    #[test]
    fn channel_split_2_to_tokens() {
        let layer = setup_2way();
        assert_eq!(
            layer.to_token_stream().to_string(),
            quote! {
                let (input, tensor_7): (
                    microflow::tensor::Tensor4D<_, 1usize, 4usize, 4usize, 16usize, 1usize>,
                    microflow::tensor::Tensor4D<_, 1usize, 4usize, 4usize, 16usize, 1usize>
                ) = microflow::ops::channel_split_2::<_, 1usize, 4usize, 4usize, 32usize, 16usize, 16usize>(input);
            }
            .to_string()
        );
    }

    #[test]
    fn channel_split_3_to_tokens() {
        let layer = setup_3way();
        assert_eq!(
            layer.to_token_stream().to_string(),
            quote! {
                let (input, tensor_6, tensor_7): (
                    microflow::tensor::Tensor4D<_, 1usize, 4usize, 4usize, 8usize, 1usize>,
                    microflow::tensor::Tensor4D<_, 1usize, 4usize, 4usize, 8usize, 1usize>,
                    microflow::tensor::Tensor4D<_, 1usize, 4usize, 4usize, 8usize, 1usize>
                ) = microflow::ops::channel_split_3::<_, 1usize, 4usize, 4usize, 24usize, 8usize, 8usize, 8usize>(input);
            }
            .to_string()
        );
    }
}
