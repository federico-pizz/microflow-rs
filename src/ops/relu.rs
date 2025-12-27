use crate::activation;
use crate::quantize::Quantized;
use crate::tensor::Tensor4D;

/// Performs the ReLU activation function as an operator.
/// Returns a 4-dimensional output tensor containing the result of the operation.
///
/// # Arguments
/// * `input` - The 4-dimensional input tensor
/// * `output_scale` - The scale of the resulting output tensor
/// * `output_zero_point` - The zero point of the resulting output tensor
///
pub fn relu<T: Quantized, const BATCHES: usize, const ROWS: usize, const COLS: usize, const CHANS: usize>(
    input: Tensor4D<T, BATCHES, ROWS, COLS, CHANS, 1>,
    output_scale: [f32; 1],
    output_zero_point: [T; 1],
) -> Tensor4D<T, BATCHES, ROWS, COLS, CHANS, 1> {
    let mut output_buffer = input.buffer;
    for batch in 0..BATCHES {
        for row in 0..ROWS {
            for col in 0..COLS {
                for chan in 0..CHANS {
                    output_buffer[batch][(row, col)][chan] = activation::relu(input.buffer[batch][(row, col)][chan], output_zero_point[0]);
                }
            }
        }
    }
    Tensor4D::new(
        output_buffer,
        output_scale,
        output_zero_point,
    )
}

/// Performs the ReLU activation function in-place on the input tensor.
///
/// # Arguments
/// * `input` - The mutable 4-dimensional input tensor to modify in place
/// * `output_scale` - The scale of the resulting output tensor
/// * `output_zero_point` - The zero point of the resulting output tensor
///
pub fn relu_in_place<T: Quantized, const BATCHES: usize, const ROWS: usize, const COLS: usize, const CHANS: usize>(
    input: &mut Tensor4D<T, BATCHES, ROWS, COLS, CHANS, 1>,
    output_scale: [f32; 1],
    output_zero_point: [T; 1],
) {
    for batch in 0..BATCHES {
        for row in 0..ROWS {
            for col in 0..COLS {
                for chan in 0..CHANS {
                    input.buffer[batch][(row, col)][chan] = activation::relu(input.buffer[batch][(row, col)][chan], output_zero_point[0]);
                }
            }
        }
    }
    input.scale = output_scale;
    input.zero_point = output_zero_point;
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::matrix;

    const INPUT: Tensor4D<i8, 1, 2, 2, 1, 1> = Tensor4D {
        buffer: [
            matrix![
                [1], [2];
                [3], [4]
            ]
        ],
        scale: [0.5],
        zero_point: [0],
    };
    const OUTPUT_SCALE: [f32; 1] = [0.5];
    const OUTPUT_ZERO_POINT: [i8; 1] = [0];
    const OUTPUT: Tensor4D<i8, 1, 2, 2, 1, 1> = Tensor4D {
        buffer: [
            matrix![
                [1], [2];
                [3], [4]
            ]
        ],
        scale: OUTPUT_SCALE,
        zero_point: OUTPUT_ZERO_POINT,
    };

    #[test]
    fn relu_layer() {
        assert_eq!(relu(INPUT, OUTPUT_SCALE, OUTPUT_ZERO_POINT), OUTPUT);
    }

    #[test]
    fn relu_in_place_layer() {
        let mut input = INPUT;
        relu_in_place(&mut input, OUTPUT_SCALE, OUTPUT_ZERO_POINT);
        assert_eq!(input, OUTPUT);
    }
}