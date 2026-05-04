use core::array;
use libm::roundf;

use simba::scalar::SupersetOf;

use crate::buffer::Buffer2D;
use crate::quantize::Quantized;
use crate::tensor::Tensor4D;

/// Performs channel-wise (axis=3) concatenation of two quantized 4D tensors.
///
/// Each input is independently requantized to the output quantization parameters.
/// The output channels are the channels of `input1` followed by the channels of `input2`.
///
/// # Arguments
/// * `input1` - First input tensor (contributes the first `CHANS1` output channels)
/// * `input2` - Second input tensor (contributes the remaining `CHANS2` output channels)
/// * `output_scale` - Scale of the output tensor
/// * `output_zero_point` - Zero point of the output tensor
/// * `constants` - Pre-computed requantization constants `((factor1, offset1), (factor2, offset2))`
///   where `factor_k = input_scale_k / output_scale` and
///   `offset_k = output_zero_point - factor_k * input_zero_point_k`
///
#[inline(always)]
pub fn concat<
    T: Quantized,
    const BATCHES: usize,
    const ROWS: usize,
    const COLS: usize,
    const CHANS1: usize,
    const CHANS2: usize,
    const OUTPUT_CHANS: usize,
>(
    input1: Tensor4D<T, BATCHES, ROWS, COLS, CHANS1, 1>,
    input2: &Tensor4D<T, BATCHES, ROWS, COLS, CHANS2, 1>,
    output_scale: [f32; 1],
    output_zero_point: [T; 1],
    constants: ((f32, f32), (f32, f32)),
) -> Tensor4D<T, BATCHES, ROWS, COLS, OUTPUT_CHANS, 1> {
    let output = array::from_fn(|b| {
        Buffer2D::from_fn(|i, j| {
            array::from_fn(|c| {
                // CHANS1 is a compile-time const; this branch is folded by the optimizer.
                let (raw, (factor, offset)) = if c < CHANS1 {
                    (
                        f32::from_subset(&input1.buffer[b][(i, j)][c]),
                        constants.0,
                    )
                } else {
                    (
                        f32::from_subset(&input2.buffer[b][(i, j)][c - CHANS1]),
                        constants.1,
                    )
                };
                T::from_superset_unchecked(&roundf(factor * raw + offset))
            })
        })
    });
    Tensor4D::new(output, output_scale, output_zero_point)
}

#[cfg(test)]
mod tests {
    use nalgebra::matrix;

    use super::*;

    // input1: 1×2×2×2, scale=0.1, zero_point=0
    // input2: 1×2×2×1, scale=0.2, zero_point=0
    // output: 1×2×2×3, scale=0.1, zero_point=0
    // factor1 = 0.1/0.1 = 1.0, offset1 = 0 - 1.0*0 = 0.0
    // factor2 = 0.2/0.1 = 2.0, offset2 = 0 - 2.0*0 = 0.0
    const INPUT1: Tensor4D<i8, 1, 2, 2, 2, 1> = Tensor4D {
        buffer: [matrix![
            [1, 2], [3, 4];
            [5, 6], [7, 8]
        ]],
        scale: [0.1],
        zero_point: [0],
    };
    const INPUT2: Tensor4D<i8, 1, 2, 2, 1, 1> = Tensor4D {
        buffer: [matrix![
            [5], [6];
            [7], [8]
        ]],
        scale: [0.2],
        zero_point: [0],
    };
    const OUTPUT_SCALE: [f32; 1] = [0.1];
    const OUTPUT_ZERO_POINT: [i8; 1] = [0];
    // output channels: [input1_ch0, input1_ch1, round(2.0 * input2_ch0)]
    const CONSTANTS: ((f32, f32), (f32, f32)) = ((1.0, 0.0), (2.0, 0.0));
    const OUTPUT: Tensor4D<i8, 1, 2, 2, 3, 1> = Tensor4D {
        buffer: [matrix![
            [1, 2, 10], [3, 4, 12];
            [5, 6, 14], [7, 8, 16]
        ]],
        scale: [0.1],
        zero_point: [0],
    };

    #[test]
    fn concat_layer() {
        assert_eq!(
            concat::<i8, 1, 2, 2, 2, 1, 3>(
                INPUT1,
                &INPUT2,
                OUTPUT_SCALE,
                OUTPUT_ZERO_POINT,
                CONSTANTS,
            ),
            OUTPUT
        );
    }
}
