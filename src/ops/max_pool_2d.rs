use core::array;
use libm::roundf;

use nalgebra::Const;
use simba::scalar::SupersetOf;

use crate::activation::{relu, relu6, FusedActivation};
use crate::buffer::Buffer2D;
use crate::quantize::Quantized;
use crate::tensor::{Tensor4D, TensorView, TensorViewPadding};

pub struct MaxPool2DOptions {
    pub fused_activation: FusedActivation,
    pub view_padding: TensorViewPadding,
    pub strides: (usize, usize),
}

/// Performs the MaxPool2D operation.
/// Returns a 4-dimensional output tensor containing the result of the operation.
///
/// # Arguments
/// * `input` - The 4-dimensional input tensor
/// * `_filter_shape` - The phantom shape of the filter
/// * `output_scale` - The scale of the resulting output tensor
/// * `output_zero_point` - The zero point of the resulting output tensor
/// * `options` - Operator's options as a [`MaxPool2DOptions`] struct
/// * `constants` - Constant values coming from the pre-processing phase
///
pub fn max_pool_2d<
    T: Quantized,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const INPUT_CHANS: usize,
    const FILTER_ROWS: usize,
    const FILTER_COLS: usize,
    const OUTPUT_ROWS: usize,
    const OUTPUT_COLS: usize,
>(
    input: Tensor4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS, 1>,
    _filter_shape: (Const<FILTER_ROWS>, Const<FILTER_COLS>),
    output_scale: [f32; 1],
    output_zero_point: [T; 1],
    options: MaxPool2DOptions,
    constants: (f32, f32),
) -> Tensor4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, INPUT_CHANS, 1> {
    let output = [Buffer2D::from_fn(|i, j| {
        // Extract the view using the view extraction algorithm
        let view: TensorView<T, FILTER_ROWS, FILTER_COLS, INPUT_CHANS> =
            input.view((i, j), 0, options.view_padding, options.strides);
        // Compute the max pooling for each channel
        array::from_fn(|c| {
            let x = view.buffer.zip_fold(&view.mask, i32::MIN, |acc, a, m| {
                if m {
                    acc.max(i32::from_subset(&a[c]))
                } else {
                    acc
                }
            }) as f32;
            let y = T::from_superset_unchecked(&roundf(constants.0 * x + constants.1));
            // Apply the fused activation function (if any)
            match options.fused_activation {
                FusedActivation::None => y,
                FusedActivation::Relu => relu(y, output_zero_point[0]),
                FusedActivation::Relu6 => relu6(y, output_scale[0], output_zero_point[0]),
            }
        })
    })];
    Tensor4D::new(output, output_scale, output_zero_point)
}

#[cfg(test)]
mod tests {
    use nalgebra::matrix;

    use super::*;

    const INPUT: Tensor4D<i8, 1, 2, 3, 2, 1> = Tensor4D {
        buffer: [matrix![
            [1, 2], [3, 4],  [5,  6];
            [7, 8], [9, 10], [11, 12]
        ]],
        scale: [0.13],
        zero_point: [14],
    };
    const FILTER_SHAPE: (Const<2>, Const<3>) = (Const, Const);
    const OUTPUT_SCALE: [f32; 1] = [0.15];
    const OUTPUT_ZERO_POINT: [i8; 1] = [16];
    const OPTIONS: MaxPool2DOptions = MaxPool2DOptions {
        fused_activation: FusedActivation::None,
        view_padding: TensorViewPadding::Same,
        strides: (1, 1),
    };
    const CONSTANTS: (f32, f32) = (0.866_666_7, 3.866_666_6);
    // Expected: max pooling over the (same-padded) 2×3 view for each output position.
    // For strides (1,1) with Same padding the view at (0,1) covers the full 2×3 input,
    // so the max picks the largest value in each channel across valid cells.
    // Channel 0 values: 1,3,5,7,9,11 → max=11; channel 1: 2,4,6,8,10,12 → max=12
    // Requantized: round(0.8667 * 11 + 3.867) = round(13.401) = 13 (ch0)
    //              round(0.8667 * 12 + 3.867) = round(14.267) = 14 (ch1)
    const OUTPUT: Tensor4D<i8, 1, 2, 3, 2, 1> = Tensor4D {
        buffer: [matrix![
            [12, 13], [13, 14], [13, 14];
            [12, 13], [13, 14], [13, 14]
        ]],
        scale: [0.15],
        zero_point: [16],
    };

    #[test]
    fn max_pool_2d_layer() {
        assert_eq!(
            max_pool_2d(
                INPUT,
                FILTER_SHAPE,
                OUTPUT_SCALE,
                OUTPUT_ZERO_POINT,
                OPTIONS,
                CONSTANTS,
            ),
            OUTPUT
        );
    }
}
