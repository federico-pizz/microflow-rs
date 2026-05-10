use core::array;
use libm::roundf;

use simba::scalar::SupersetOf;

use crate::buffer::Buffer2D;
use crate::quantize::Quantized;
use crate::tensor::Tensor4D;

/// Performs ReduceMean over the spatial dimensions (H and W) of a quantized 4D tensor,
/// producing a (BATCHES, 1, 1, CHANS) output — equivalent to global average pooling.
/// Returns a 4-dimensional output tensor containing the result of the operation.
///
/// # Arguments
/// * `input` - The 4-dimensional input tensor with shape (BATCHES, ROWS, COLS, CHANS)
/// * `output_scale` - The scale of the resulting output tensor
/// * `output_zero_point` - The zero point of the resulting output tensor
/// * `constants` - Pre-computed values `(c0, c1)` where
///   `c0 = input_scale / (output_scale * ROWS * COLS)` and `c1 = f32(output_zero_point)`
///
/// Formula: `output[b,c] = round(c0 * sum_{r,col}(input[b,r,col,c] - input_zp*R*C) + c1)`
///
#[inline(always)]
pub fn reduce_mean<
    T: Quantized,
    const BATCHES: usize,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const INPUT_CHANS: usize,
>(
    input: Tensor4D<T, BATCHES, INPUT_ROWS, INPUT_COLS, INPUT_CHANS, 1>,
    output_scale: [f32; 1],
    output_zero_point: [T; 1],
    constants: (f32, f32),
) -> Tensor4D<T, BATCHES, 1, 1, INPUT_CHANS, 1> {
    let input_zp = i32::from_subset(&input.zero_point[0]);
    let n = (INPUT_ROWS * INPUT_COLS) as i32;
    let output: [Buffer2D<[T; INPUT_CHANS], 1, 1>; BATCHES] = array::from_fn(|b| {
        Buffer2D::from_fn(|_, _| {
            let n_times_zp = n * input_zp;
            array::from_fn(|c| {
                // Accumulate integer sum of raw quantized values for this channel
                let sum = (0..INPUT_ROWS).fold(0i32, |acc, r| {
                    (0..INPUT_COLS).fold(acc, |acc, col| {
                        acc + i32::from_subset(&input.buffer[b][(r, col)][c])
                    })
                });
                // Subtract n*input_zp to convert sum-of-quantized to sum-of-dequantized (without scale)
                // constants.0 = input_scale / (output_scale * n), constants.1 = f32(output_zero_point)
                T::from_superset_unchecked(&roundf(
                    constants.0 * (sum - n_times_zp) as f32 + constants.1,
                ))
            })
        })
    });
    Tensor4D::new(output, output_scale, output_zero_point)
}

#[cfg(test)]
mod tests {
    use nalgebra::matrix;

    use super::*;

    // input: shape (1, 2, 3, 2), scale=0.1, zero_point=0
    // reduce over 2×3=6 elements per channel
    // channel 0 values: 1,3,5,7,9,11 → mean=6.0 → real=0.6
    // channel 1 values: 2,4,6,8,10,12 → mean=7.0 → real=0.7
    // output: scale=0.1, zero_point=0
    // c0 = 0.1 / (0.1 * 6) = 1/6, c1 = 0.0
    // out[0] = round(1/6 * 36) = round(6.0) = 6
    // out[1] = round(1/6 * 42) = round(7.0) = 7
    const INPUT: Tensor4D<i8, 1, 2, 3, 2, 1> = Tensor4D {
        buffer: [matrix![
            [1, 2], [3, 4],  [5,  6];
            [7, 8], [9, 10], [11, 12]
        ]],
        scale: [0.1],
        zero_point: [0],
    };
    const OUTPUT_SCALE: [f32; 1] = [0.1];
    const OUTPUT_ZERO_POINT: [i8; 1] = [0];
    // c0 = input_scale / (output_scale * R * C) = 0.1 / (0.1 * 6) ≈ 0.166667
    // c1 = f32(output_zero_point) = 0.0
    const CONSTANTS: (f32, f32) = (1.0 / 6.0, 0.0);
    const OUTPUT: Tensor4D<i8, 1, 1, 1, 2, 1> = Tensor4D {
        buffer: [matrix![[6, 7]]],
        scale: [0.1],
        zero_point: [0],
    };

    #[test]
    fn reduce_mean_layer() {
        assert_eq!(
            reduce_mean(INPUT, OUTPUT_SCALE, OUTPUT_ZERO_POINT, CONSTANTS),
            OUTPUT
        );
    }
}
