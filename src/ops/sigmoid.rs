use crate::activation;
use crate::quantize::Quantized;
use crate::quantize::dequantize;
use crate::tensor::Tensor4D;
use crate::buffer::Buffer2D;
use core::array;

/// Performs the Sigmoid activation function as an operator on a quantized 4D tensor.
/// Returns a 4-dimensional output tensor containing the result of the operation.
///
/// The output scale and zero point for int8 TFLite models are typically fixed at
/// 1/256 and -128 respectively, but this implementation accepts arbitrary values.
///
/// # Arguments
/// * `input` - The 4-dimensional input tensor
/// * `output_scale` - The scale of the resulting output tensor
/// * `output_zero_point` - The zero point of the resulting output tensor
///
#[inline(always)]
pub fn sigmoid<
    T: Quantized,
    const BATCHES: usize,
    const ROWS: usize,
    const COLS: usize,
    const CHANS: usize,
>(
    input: Tensor4D<T, BATCHES, ROWS, COLS, CHANS, 1>,
    output_scale: [f32; 1],
    output_zero_point: [T; 1],
) -> Tensor4D<T, BATCHES, ROWS, COLS, CHANS, 1> {
    let in_scale = input.scale[0];
    let in_zp = input.zero_point[0];
    let output = array::from_fn(|b| {
        Buffer2D::from_fn(|i, j| {
            array::from_fn(|c| {
                let x = dequantize(input.buffer[b][(i, j)][c], in_scale, in_zp);
                activation::sigmoid(x, output_scale[0], output_zero_point[0])
            })
        })
    });
    Tensor4D::new(output, output_scale, output_zero_point)
}

#[cfg(test)]
mod tests {
    use nalgebra::matrix;

    use super::*;

    // input: scale=1.0, zero_point=0
    // so dequantized values are [-2, -1, 0, 1] etc.
    // sigmoid(0) = 0.5, sigmoid(1) ≈ 0.731, sigmoid(-1) ≈ 0.269, sigmoid(2) ≈ 0.880
    // output: scale=1/256≈0.003906, zero_point=-128
    // quantized = round(sigmoid(x) / 0.003906) + (-128)
    // sigmoid(0)=0.5 → round(0.5/0.003906)-128 = 128-128 = 0
    // sigmoid(1)≈0.731 → round(0.731/0.003906)-128 ≈ 187-128 = 59
    // sigmoid(-1)≈0.269 → round(0.269/0.003906)-128 ≈ 69-128 = -59
    // sigmoid(2)≈0.880 → round(0.880/0.003906)-128 ≈ 225-128 = 97
    const INPUT: Tensor4D<i8, 1, 2, 2, 1, 1> = Tensor4D {
        buffer: [matrix![
            [0], [1];
            [-1], [2]
        ]],
        scale: [1.0],
        zero_point: [0],
    };
    const OUTPUT_SCALE: [f32; 1] = [1.0 / 256.0];
    const OUTPUT_ZERO_POINT: [i8; 1] = [-128];
    const OUTPUT: Tensor4D<i8, 1, 2, 2, 1, 1> = Tensor4D {
        buffer: [matrix![
            [0],  [59];
            [-59], [97]
        ]],
        scale: [1.0 / 256.0],
        zero_point: [-128],
    };

    #[test]
    fn sigmoid_layer() {
        assert_eq!(sigmoid(INPUT, OUTPUT_SCALE, OUTPUT_ZERO_POINT), OUTPUT);
    }
}
