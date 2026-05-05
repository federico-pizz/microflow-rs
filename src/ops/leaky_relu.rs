use crate::activation;
use crate::quantize::{dequantize, Quantized};
use crate::tensor::Tensor4D;

/// Performs the LeakyReLU activation function as an operator on a quantized 4D tensor.
///
/// LeakyReLU: alpha * x for x < 0, x otherwise.
#[inline(always)]
pub fn leaky_relu<
    T: Quantized,
    const BATCHES: usize,
    const ROWS: usize,
    const COLS: usize,
    const CHANS: usize,
>(
    mut input: Tensor4D<T, BATCHES, ROWS, COLS, CHANS, 1>,
    alpha: f32,
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
                        activation::leaky_relu(x, alpha, output_scale[0], output_zero_point[0]);
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

    // alpha = 0.1
    // leaky_relu(-2) = 0.1 * -2 = -0.2, leaky_relu(0)=0, leaky_relu(1)=1, leaky_relu(3)=3
    // output: scale=0.1, zero_point=0
    // quantized = round(lr(x) / 0.1)
    // lr(-2)=-0.2 → -2
    // lr(0)=0 → 0
    // lr(1)=1 → 10
    // lr(3)=3 → 30
    const INPUT: Tensor4D<i8, 1, 2, 2, 1, 1> = Tensor4D {
        buffer: [matrix![
            [-20], [0];
            [10], [30]
        ]],
        scale: [0.1],
        zero_point: [0],
    };
    const ALPHA: f32 = 0.1;
    const OUTPUT_SCALE: [f32; 1] = [0.1];
    const OUTPUT_ZERO_POINT: [i8; 1] = [0];
    const OUTPUT: Tensor4D<i8, 1, 2, 2, 1, 1> = Tensor4D {
        buffer: [matrix![
            [-2], [0];
            [10], [30]
        ]],
        scale: [0.1],
        zero_point: [0],
    };

    #[test]
    fn leaky_relu_layer() {
        assert_eq!(leaky_relu(INPUT, ALPHA, OUTPUT_SCALE, OUTPUT_ZERO_POINT), OUTPUT);
    }
}
