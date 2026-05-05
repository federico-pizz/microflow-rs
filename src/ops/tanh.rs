use crate::activation;
use crate::quantize::{dequantize, Quantized};
use crate::tensor::Tensor4D;

/// Performs the Tanh activation function as an operator on a quantized 4D tensor.
#[inline(always)]
pub fn tanh<
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
                        activation::tanh(x, output_scale[0], output_zero_point[0]);
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

    // tanh(0) = 0.0, tanh(1) ≈ 0.7616, tanh(-1) ≈ -0.7616, tanh(2) ≈ 0.9640
    // output: scale=1/128≈0.007812, zero_point=0
    // quantized = round(tanh(x) / 0.007812)
    // tanh(0)=0 → 0
    // tanh(1)≈0.7616 → round(0.7616/0.007812) = 97
    // tanh(-1)≈-0.7616 → -97
    // tanh(2)≈0.9640 → round(0.9640/0.007812) = 123
    const INPUT: Tensor4D<i8, 1, 2, 2, 1, 1> = Tensor4D {
        buffer: [matrix![
            [0], [1];
            [-1], [2]
        ]],
        scale: [1.0],
        zero_point: [0],
    };
    const OUTPUT_SCALE: [f32; 1] = [1.0 / 128.0];
    const OUTPUT_ZERO_POINT: [i8; 1] = [0];
    const OUTPUT: Tensor4D<i8, 1, 2, 2, 1, 1> = Tensor4D {
        buffer: [matrix![
            [0], [97];
            [-97], [123]
        ]],
        scale: [1.0 / 128.0],
        zero_point: [0],
    };

    #[test]
    fn tanh_layer() {
        assert_eq!(tanh(INPUT, OUTPUT_SCALE, OUTPUT_ZERO_POINT), OUTPUT);
    }
}
