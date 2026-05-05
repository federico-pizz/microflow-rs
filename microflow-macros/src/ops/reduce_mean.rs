use std::mem::size_of;

use flatbuffers::{ForwardsUOffset, Vector};
use proc_macro2::TokenStream as TokenStream2;
use quote::{quote, ToTokens};
use simba::scalar::SupersetOf;

use crate::quantize::TokenQuantized;
use crate::tensor::TokenTensor4D;
use crate::tflite_flatbuffers::tflite::{Buffer, Operator, Tensor, TensorType};

/// Represents the tokenized version of the `ReduceMean` (MEAN) operator over H and W.
pub(crate) struct TokenReduceMean<T: TokenQuantized> {
    pub(crate) output: TokenTensor4D<T>,
    /// (c0, c1) where c0 = in_scale / (out_scale * R * C), c1 = f32(out_zero_point).
    pub(crate) constants: (f32, f32),
}

pub(crate) fn parse(
    operator: Operator,
    tensors: Vector<ForwardsUOffset<Tensor>>,
    buffers: Vector<ForwardsUOffset<Buffer>>,
) -> Box<dyn ToTokens> {
    let inputs = operator.inputs().unwrap();
    let input_type = tensors.get(inputs.get(0) as usize).type_();
    match input_type {
        TensorType::INT8 => Box::new(TokenReduceMean::<i8>::new(operator, tensors, buffers)),
        TensorType::UINT8 => Box::new(TokenReduceMean::<u8>::new(operator, tensors, buffers)),
        _ => unimplemented!(),
    }
}

impl<T: TokenQuantized> TokenReduceMean<T> {
    pub(crate) fn new(
        operator: Operator,
        tensors: Vector<ForwardsUOffset<Tensor>>,
        buffers: Vector<ForwardsUOffset<Buffer>>,
    ) -> Self {
        let inputs = operator.inputs().unwrap();
        let input = TokenTensor4D::<T>::from_empty_tensor(tensors.get(inputs.get(0) as usize));

        // Read axes from the constant tensor (inputs[1]) and assert H+W reduction.
        let axes_tensor = tensors.get(inputs.get(1) as usize);
        let axes_bytes = buffers
            .get(axes_tensor.buffer() as usize)
            .data()
            .unwrap()
            .bytes();
        let n_axes = axes_bytes.len() / size_of::<i32>();
        let axes: Vec<i32> = (0..n_axes)
            .map(|i| {
                i32::from_le_bytes(
                    axes_bytes[i * size_of::<i32>()..(i + 1) * size_of::<i32>()]
                        .try_into()
                        .unwrap(),
                )
            })
            .collect();
        assert_eq!(
            axes,
            vec![1, 2],
            "only H+W (axes=[1,2]) ReduceMean is supported, got axes={:?}",
            axes
        );

        let output = TokenTensor4D::<T>::from_empty_tensor(
            tensors.get(operator.outputs().unwrap().get(0) as usize),
        );
        let constants = Self::preprocess(&input, &output);
        Self { output, constants }
    }

    fn preprocess(input: &TokenTensor4D<T>, output: &TokenTensor4D<T>) -> (f32, f32) {
        let rows = input.shape[1] as f32;
        let cols = input.shape[2] as f32;
        let c0 = input.scale[0] / (output.scale[0] * rows * cols);
        let c1 = f32::from_subset(&output.zero_point[0]);
        (c0, c1)
    }
}

impl<T: TokenQuantized> ToTokens for TokenReduceMean<T> {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let output_shape = &self.output.shape;
        let output_scale = &self.output.scale;
        let output_zero_point = &self.output.zero_point;
        let (c0, c1) = self.constants;

        let ts = quote! {
            let input: microflow::tensor::Tensor4D<_, #(#output_shape),*, 1usize> =
                microflow::ops::reduce_mean(
                    input,
                    [#(#output_scale),*], [#(#output_zero_point),*],
                    (#c0, #c1),
                );
        };
        ts.to_tokens(tokens);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::TokenBuffer4D;

    fn setup() -> TokenReduceMean<i8> {
        TokenReduceMean {
            output: TokenTensor4D {
                buffer: TokenBuffer4D::new(),
                shape: vec![1, 1, 1, 8],
                scale: vec![0.1],
                zero_point: vec![0i8],
            },
            constants: (1.0 / 16.0, 0.0),
        }
    }

    #[test]
    fn reduce_mean_to_tokens() {
        let layer = setup();
        assert_eq!(
            layer.to_token_stream().to_string(),
            quote! {
                let input: microflow::tensor::Tensor4D<_, 1usize, 1usize, 1usize, 8usize, 1usize> =
                    microflow::ops::reduce_mean(
                        input,
                        [0.1f32], [0i8],
                        (0.0625f32, 0f32),
                    );
            }
            .to_string()
        );
    }
}
