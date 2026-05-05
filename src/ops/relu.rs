use crate::quantize::{dequantize, quantize, Quantized};
use crate::tensor::Tensor4D;

/// Applies ReLU as a standalone operator with optional requantization.
#[inline(always)]
pub fn relu<
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
                    let y = if x > 0.0 { x } else { 0.0 };
                    input.buffer[b][(i, j)][c] =
                        quantize(y, output_scale[0], output_zero_point[0]);
                }
            }
        }
    }
    input.scale = output_scale;
    input.zero_point = output_zero_point;
    input
}

/// Applies ReLU6 as a standalone operator with optional requantization.
#[inline(always)]
pub fn relu6<
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
                    let y = x.clamp(0.0, 6.0);
                    input.buffer[b][(i, j)][c] =
                        quantize(y, output_scale[0], output_zero_point[0]);
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

    // relu: input scale=1.0, zp=0; output scale=0.5, zp=0
    // relu(-2)=0→0, relu(0)=0→0, relu(1)=1→2, relu(3)=3→6
    const INPUT: Tensor4D<i8, 1, 2, 2, 1, 1> = Tensor4D {
        buffer: [matrix![
            [-2], [0];
            [1], [3]
        ]],
        scale: [1.0],
        zero_point: [0],
    };
    const OUTPUT_SCALE: [f32; 1] = [0.5];
    const OUTPUT_ZERO_POINT: [i8; 1] = [0];
    const RELU_OUTPUT: Tensor4D<i8, 1, 2, 2, 1, 1> = Tensor4D {
        buffer: [matrix![
            [0], [0];
            [2], [6]
        ]],
        scale: [0.5],
        zero_point: [0],
    };

    // relu6: input scale=1.0, zp=0; output scale=0.5, zp=0
    // relu6(-2)=0→0, relu6(0)=0→0, relu6(3)=3→6, relu6(8)=6→12
    const INPUT6: Tensor4D<i8, 1, 2, 2, 1, 1> = Tensor4D {
        buffer: [matrix![
            [-2], [0];
            [3], [8]
        ]],
        scale: [1.0],
        zero_point: [0],
    };
    const RELU6_OUTPUT: Tensor4D<i8, 1, 2, 2, 1, 1> = Tensor4D {
        buffer: [matrix![
            [0], [0];
            [6], [12]
        ]],
        scale: [0.5],
        zero_point: [0],
    };

    #[test]
    fn relu_layer() {
        assert_eq!(relu(INPUT, OUTPUT_SCALE, OUTPUT_ZERO_POINT), RELU_OUTPUT);
    }

    #[test]
    fn relu6_layer() {
        assert_eq!(relu6(INPUT6, OUTPUT_SCALE, OUTPUT_ZERO_POINT), RELU6_OUTPUT);
    }
}
