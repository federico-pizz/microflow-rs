use std::mem::size_of;

use flatbuffers::{ForwardsUOffset, Vector};
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote, ToTokens};

use crate::quantize::TokenQuantized;
use crate::tensor::TokenTensor4D;
use crate::tflite_flatbuffers::tflite::{Buffer, Operator, Tensor};

/// Represents the tokenized version of a 2-way channel split operator.
pub(crate) struct TokenSplit<T: TokenQuantized> {
    /// Primary output (first half of channels) — becomes the new `input` variable.
    pub(crate) output1: TokenTensor4D<T>,
    /// Secondary output (second half of channels) — saved as a named variable.
    pub(crate) output2: TokenTensor4D<T>,
    /// Tensor index of the secondary output, used to form the variable name `tensor_N`.
    pub(crate) secondary_idx: usize,
}

/// Parses a [`TokenSplit`] from the given TFLite SPLIT operator.
///
/// # Arguments
/// * `operator` - The model operator
/// * `tensors`  - All tensors in the subgraph
/// * `buffers`  - All buffers in the model (needed to read the axis constant)
pub(crate) fn parse<T: TokenQuantized>(
    operator: Operator,
    tensors: Vector<ForwardsUOffset<Tensor>>,
    buffers: Vector<ForwardsUOffset<Buffer>>,
) -> TokenSplit<T> {
    TokenSplit::<T>::new(operator, tensors, buffers)
}

impl<T: TokenQuantized> TokenSplit<T> {
    pub(crate) fn new(
        operator: Operator,
        tensors: Vector<ForwardsUOffset<Tensor>>,
        buffers: Vector<ForwardsUOffset<Buffer>>,
    ) -> Self {
        let inputs = operator.inputs().unwrap();
        let outputs = operator.outputs().unwrap();

        let options = operator.builtin_options_as_split_options().unwrap();
        assert_eq!(
            options.num_splits(),
            2,
            "only 2-way SPLIT is supported, got {} splits",
            options.num_splits()
        );
        assert_eq!(outputs.len(), 2, "expected 2 outputs for SPLIT");

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

        let output1_idx = outputs.get(0) as usize;
        let output2_idx = outputs.get(1) as usize;
        let output1 = TokenTensor4D::<T>::from_empty_tensor(tensors.get(output1_idx));
        let output2 = TokenTensor4D::<T>::from_empty_tensor(tensors.get(output2_idx));

        Self {
            output1,
            output2,
            secondary_idx: output2_idx,
        }
    }
}

impl<T: TokenQuantized> ToTokens for TokenSplit<T> {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let saved_ident = format_ident!("tensor_{}", self.secondary_idx);

        let out1_shape = &self.output1.shape;
        let out2_shape = &self.output2.shape;

        // Total input channel count = CHANS1 + CHANS2.  The compiler requires it to be
        // explicit in the turbofish so it can verify the buffer layout at compile time.
        let total_chans = out1_shape[3] + out2_shape[3];
        let chans1 = out1_shape[3];
        let chans2 = out2_shape[3];
        let batches = out1_shape[0];
        let rows = out1_shape[1];
        let cols = out1_shape[2];

        let ts = quote! {
            let (input, #saved_ident): (
                microflow::tensor::Tensor4D<_, #batches, #rows, #cols, #chans1, 1usize>,
                microflow::tensor::Tensor4D<_, #batches, #rows, #cols, #chans2, 1usize>,
            ) = microflow::ops::split_2::<_, #batches, #rows, #cols, #total_chans, #chans1, #chans2>(input);
        };
        ts.to_tokens(tokens);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::TokenBuffer4D;

    fn setup() -> TokenSplit<i8> {
        TokenSplit {
            output1: TokenTensor4D {
                buffer: TokenBuffer4D::new(),
                shape: vec![1, 4, 4, 16],
                scale: vec![0.1],
                zero_point: vec![0i8],
            },
            output2: TokenTensor4D {
                buffer: TokenBuffer4D::new(),
                shape: vec![1, 4, 4, 16],
                scale: vec![0.1],
                zero_point: vec![0i8],
            },
            secondary_idx: 7,
        }
    }

    #[test]
    fn split_to_tokens() {
        let layer = setup();
        assert_eq!(
            layer.to_token_stream().to_string(),
            quote! {
                let (input, tensor_7): (
                    microflow::tensor::Tensor4D<_, 1usize, 4usize, 4usize, 16usize, 1usize>,
                    microflow::tensor::Tensor4D<_, 1usize, 4usize, 4usize, 16usize, 1usize>,
                ) = microflow::ops::split_2::<_, 1usize, 4usize, 4usize, 32usize, 16usize, 16usize>(input);
            }
            .to_string()
        );
    }
}
