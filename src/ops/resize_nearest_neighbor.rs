use crate::buffer::Buffer2D;
use crate::quantize::Quantized;
use crate::tensor::Tensor4D;
use libm::{floorf, roundf};
use simba::scalar::SupersetOf;

#[derive(Clone, Copy, Default)]
pub struct ResizeNearestNeighborOptions {
    pub align_corners: bool,
    pub half_pixel_centers: bool,
}

/// Performs the ResizeNearestNeighbor operation.
/// Returns a 4-dimensional output tensor containing the resized input.
///
/// # Arguments
/// * `input` - The 4-dimensional input tensor passed by value.
/// * `output_scale` - The scale of the resulting output tensor.
/// * `output_zero_point` - The zero point of the resulting output tensor.
/// * `options` - Operator's options as a [`ResizeNearestNeighborOptions`] struct.
/// * `constants` - A tuple of pre-calculated f32 values for requantization.
pub fn resize_nearest_neighbor<
    T: Quantized,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const INPUT_CHANS: usize,
    const OUTPUT_ROWS: usize,
    const OUTPUT_COLS: usize,
>(
    input: Tensor4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS, 1>,
    output_scale: [f32; 1],
    output_zero_point: [T; 1],
    options: ResizeNearestNeighborOptions,
    constants: (f32, f32),
) -> Tensor4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, INPUT_CHANS, 1> {
    let output = [Buffer2D::from_fn(|r, c| {
        // Calculate the corresponding row and column in the input tensor
        let in_r = if options.half_pixel_centers {
            ((r as f32 + 0.5) * (INPUT_ROWS as f32 / OUTPUT_ROWS as f32)) as usize
        } else if options.align_corners && OUTPUT_ROWS > 1 {
            roundf(r as f32 * (INPUT_ROWS - 1) as f32 / (OUTPUT_ROWS - 1) as f32) as usize
        } else {
            floorf(r as f32 * INPUT_ROWS as f32 / OUTPUT_ROWS as f32) as usize
        };

        let in_c = if options.half_pixel_centers {
            ((c as f32 + 0.5) * (INPUT_COLS as f32 / OUTPUT_COLS as f32)) as usize
        } else if options.align_corners && OUTPUT_COLS > 1 {
            roundf(c as f32 * (INPUT_COLS - 1) as f32 / (OUTPUT_COLS - 1) as f32) as usize
        } else {
            floorf(c as f32 * INPUT_COLS as f32 / OUTPUT_COLS as f32) as usize
        };

        // Ensure indices are within bounds
        let nearest_r = in_r.min(INPUT_ROWS.saturating_sub(1));
        let nearest_c = in_c.min(INPUT_COLS.saturating_sub(1));

        let mut out_channels = [T::from_superset_unchecked(&0); INPUT_CHANS];
        for i in 0..INPUT_CHANS {
            let p = f32::from_subset(&input.buffer[0][(nearest_r, nearest_c)][i]);

            // Requantize the output value
            let requantized = roundf(constants.0 * p + constants.1);

            // Validate for NaN/infinity and use safe conversion
            let clamped = if requantized.is_nan() || requantized.is_infinite() {
                0.0f32 // Default to zero for invalid results
            } else {
                requantized
            };

            // Use safe conversion with fallback to zero
            out_channels[i] = T::from_superset(&clamped).unwrap_or_else(|| T::from_superset_unchecked(&0.0f32));
        }
        out_channels
    })];
    Tensor4D::new(output, output_scale, output_zero_point)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::matrix;

    const INPUT: Tensor4D<i8, 1, 2, 2, 1, 1> = Tensor4D {
        buffer: [matrix![[10], [20]; [30], [40]]],
        scale: [1.0],
        zero_point: [0],
    };
    const OUTPUT_SCALE: [f32; 1] = [1.0];
    const OUTPUT_ZERO_POINT: [i8; 1] = [0];
    const CONSTANTS: (f32, f32) = (1.0, 0.0);

    #[test]
    fn test_upscaling() {
        let options = ResizeNearestNeighborOptions {
            align_corners: false,
            half_pixel_centers: false,
        };
        let output: Tensor4D<i8, 1, 4, 4, 1, 1> = resize_nearest_neighbor(
            INPUT,
            OUTPUT_SCALE,
            OUTPUT_ZERO_POINT,
            options,
            CONSTANTS,
        );
        let expected = [
            [10, 10, 20, 20],
            [10, 10, 20, 20],
            [30, 30, 40, 40],
            [30, 30, 40, 40],
        ];
        for r in 0..4 {
            for c in 0..4 {
                assert_eq!(output.buffer[0][(r, c)][0], expected[r][c]);
            }
        }
    }

    #[test]
    fn test_downscaling() {
        let input: Tensor4D<i8, 1, 4, 4, 1, 1> = Tensor4D {
            buffer: [matrix![
                [10], [20], [30], [40];
                [50], [60], [70], [80];
                [90], [100], [110], [120];
                [127], [127], [127], [127]
            ]],
            scale: [1.0],
            zero_point: [0],
        };
        let options = ResizeNearestNeighborOptions {
            align_corners: false,
            half_pixel_centers: false,
        };
        let output: Tensor4D<i8, 1, 2, 2, 1, 1> = resize_nearest_neighbor(
            input,
            OUTPUT_SCALE,
            OUTPUT_ZERO_POINT,
            options,
            CONSTANTS,
        );
        assert_eq!(output.buffer[0][(0, 0)][0], 10);
        assert_eq!(output.buffer[0][(0, 1)][0], 30);
        assert_eq!(output.buffer[0][(1, 0)][0], 90);
        assert_eq!(output.buffer[0][(1, 1)][0], 110);
    }
    
    #[test]
    fn test_align_corners() {
        let options = ResizeNearestNeighborOptions {
            align_corners: true,
            half_pixel_centers: false,
        };
        let output: Tensor4D<i8, 1, 3, 3, 1, 1> = resize_nearest_neighbor(
            INPUT,
            OUTPUT_SCALE,
            OUTPUT_ZERO_POINT,
            options,
            CONSTANTS,
        );
        let expected = [[10, 20, 20], [30, 40, 40], [30, 40, 40]];
        for r in 0..3 {
            for c in 0..3 {
                assert_eq!(output.buffer[0][(r, c)][0], expected[r][c]);
            }
        }
    }

    #[test]
    fn test_half_pixel_centers() {
        let options = ResizeNearestNeighborOptions {
            align_corners: false,
            half_pixel_centers: true,
        };
        let output: Tensor4D<i8, 1, 3, 3, 1, 1> = resize_nearest_neighbor(
            INPUT,
            OUTPUT_SCALE,
            OUTPUT_ZERO_POINT,
            options,
            CONSTANTS,
        );
        assert_eq!(output.buffer[0][(1, 1)][0], 40); 
    }
}
