use crate::activation;
use crate::quantize::{dequantize, Quantized};
use crate::tensor::Tensor4D;

/// Performs the HardSwish activation function as an operator on a quantized 4D tensor.
///
/// HardSwish: x * relu6(x + 3) / 6
#[inline(always)]
pub fn hard_swish<
    T: Quantized,
    const BATCHES: usize,
    const ROWS: usize,
    const COLS: usize,
    const CHANS: usize,
>(
    mut input: Tensor4D<T, BATCHES, ROWS, COLS, CHANS, 1>,
    output_scale: [f32; 1],
    output_zero_point: [T; 1],
) -> Tensor4D<T, BATCHES, ROWS, COLS, CHANS, 1> {
    let in_scale = input.scale[0];
    let in_zp = input.zero_point[0];
    for b in 0..BATCHES {
        for i in 0..ROWS {
            for j in 0..COLS {
                let vals = input.buffer[b][(i, j)];
                for c in 0..CHANS {
                    let x = dequantize(vals[c], in_scale, in_zp);
                    input.buffer[b][(i, j)][c] =
                        activation::hard_swish(x, output_scale[0], output_zero_point[0]);
                }
            }
        }
    }
    input.scale = output_scale;
    input.zero_point = output_zero_point;
    input
}

#[cfg(test)]
mod tests {
    use nalgebra::matrix;

    use super::*;

    // hard_swish(-4) = 0, hard_swish(0) = 0, hard_swish(1) = 1*(1+3)/6 = 4/6 ≈ 0.667, hard_swish(4) = 4
    // output: scale=0.1, zero_point=0
    // quantized = round(hs(x) / 0.1)
    // hs(-4)=0 → 0
    // hs(0)=0 → 0
    // hs(1)≈0.667 → 7
    // hs(4)=4 → 40 (clamped to i8 max=127)
    const INPUT: Tensor4D<i8, 1, 2, 2, 1, 1> = Tensor4D {
        buffer: [matrix![
            [-40], [0];
            [10], [40]
        ]],
        scale: [0.1],
        zero_point: [0],
    };
    const OUTPUT_SCALE: [f32; 1] = [0.1];
    const OUTPUT_ZERO_POINT: [i8; 1] = [0];
    const OUTPUT: Tensor4D<i8, 1, 2, 2, 1, 1> = Tensor4D {
        buffer: [matrix![
            [0], [0];
            [7], [40]
        ]],
        scale: [0.1],
        zero_point: [0],
    };

    #[test]
    fn hard_swish_layer() {
        assert_eq!(hard_swish(INPUT, OUTPUT_SCALE, OUTPUT_ZERO_POINT), OUTPUT);
    }
}
