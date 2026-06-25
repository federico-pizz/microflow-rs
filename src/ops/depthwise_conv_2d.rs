use core::array;

use simba::scalar::SupersetOf;

use crate::activation::{relu, relu6, FusedActivation};
use crate::buffer::Buffer2D;
use crate::quantize::{multiply_by_quantized_multiplier, quantize, saturating_cast, Quantized};
use crate::tensor::{Tensor4D, TensorView, TensorViewPadding};

pub struct DepthwiseConv2DOptions {
    pub fused_activation: FusedActivation,
    pub view_padding: TensorViewPadding,
    pub strides: (usize, usize),
}

/// Performs the DepthwiseConv2D operation.
/// Returns a 4-dimensional output tensor containing the result of the operation.
///
/// # Arguments
/// * `input` - The 4-dimensional input tensor
/// * `weights` - The 4-dimensional tensor representing the weights of the operator
/// * `output_scale` - The scale of the resulting output tensor
/// * `output_zero_point` - The zero point of the resulting output tensor
/// * `options` - Operator's options as an [`DepthwiseConv2DOptions`] struct
/// * `constants` - Constant values coming from the pre-processing phase
///
#[inline(always)]
pub fn depthwise_conv_2d<
    T: Quantized,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const INPUT_CHANS: usize,
    const WEIGHTS_ROWS: usize,
    const WEIGHTS_COLS: usize,
    const WEIGHTS_CHANS: usize,
    const WEIGHTS_QUANTS: usize,
    const OUTPUT_ROWS: usize,
    const OUTPUT_COLS: usize,
>(
    input: Tensor4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS, 1>,
    weights: &Tensor4D<T, 1, WEIGHTS_ROWS, WEIGHTS_COLS, WEIGHTS_CHANS, WEIGHTS_QUANTS>,
    output_scale: [f32; 1],
    output_zero_point: [T; 1],
    options: DepthwiseConv2DOptions,
    constants: (
        Buffer2D<i32, WEIGHTS_CHANS, 1>,
        Buffer2D<i32, WEIGHTS_QUANTS, 1>,
        Buffer2D<i32, WEIGHTS_QUANTS, 1>,
        Buffer2D<i32, WEIGHTS_CHANS, 1>,
    ),
) -> Tensor4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, WEIGHTS_CHANS, 1> {
    // Thin by-value wrapper around the in-place variant. Used by host unit tests
    // and by the non-`buffer-reuse` codegen path.
    let mut output = Tensor4D::new(
        [Buffer2D::from_element([output_zero_point[0]; WEIGHTS_CHANS])],
        output_scale,
        output_zero_point,
    );
    depthwise_conv_2d_into(
        &mut output,
        &input,
        weights,
        output_scale,
        output_zero_point,
        options,
        constants,
    );
    output
}

/// Performs the DepthwiseConv2D operation in place, writing the result into a
/// caller-owned output tensor instead of returning a fresh one.
///
/// See [`crate::ops::conv_2d::conv_2d_into`] for why the in-place form is the
/// buffer-reuse entry point.
///
/// # Arguments
/// * `output` - The 4-dimensional output tensor to fill
/// * `input` - The 4-dimensional input tensor
/// * `weights` - The 4-dimensional tensor representing the weights of the operator
/// * `output_scale` - The scale of the resulting output tensor
/// * `output_zero_point` - The zero point of the resulting output tensor
/// * `options` - Operator's options as an [`DepthwiseConv2DOptions`] struct
/// * `constants` - Constant values coming from the pre-processing phase
///
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn depthwise_conv_2d_into<
    T: Quantized,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const INPUT_CHANS: usize,
    const WEIGHTS_ROWS: usize,
    const WEIGHTS_COLS: usize,
    const WEIGHTS_CHANS: usize,
    const WEIGHTS_QUANTS: usize,
    const OUTPUT_ROWS: usize,
    const OUTPUT_COLS: usize,
>(
    output: &mut Tensor4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, WEIGHTS_CHANS, 1>,
    input: &Tensor4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS, 1>,
    weights: &Tensor4D<T, 1, WEIGHTS_ROWS, WEIGHTS_COLS, WEIGHTS_CHANS, WEIGHTS_QUANTS>,
    output_scale: [f32; 1],
    output_zero_point: [T; 1],
    options: DepthwiseConv2DOptions,
    constants: (
        Buffer2D<i32, WEIGHTS_CHANS, 1>,
        Buffer2D<i32, WEIGHTS_QUANTS, 1>,
        Buffer2D<i32, WEIGHTS_QUANTS, 1>,
        Buffer2D<i32, WEIGHTS_CHANS, 1>,
    ),
) {
    output.scale = output_scale;
    output.zero_point = output_zero_point;
    let buffer = &mut output.buffer[0];
    for j in 0..OUTPUT_COLS {
        for i in 0..OUTPUT_ROWS {
            // Extract the view using the view extraction algorithm
            let view: TensorView<T, WEIGHTS_ROWS, WEIGHTS_COLS, INPUT_CHANS> =
                input.view((i, j), 0, options.view_padding, options.strides);
            // Perform the convolution for each input channel
            buffer[(i, j)] = array::from_fn(|c| {
                let relu6_max = quantize(6., output_scale[0], output_zero_point[0]);
                let input_zero_point = i32::from_subset(&input.zero_point[0]);
                let output_zero_point_i32 = i32::from_subset(&output_zero_point[0]);
                let weights_zero_point = i32::from_subset(
                    &weights
                        .zero_point
                        .get(c)
                        .copied()
                        .unwrap_or(weights.zero_point[0]),
                );
                // Single pass: accumulate MAC and per-channel view sum simultaneously
                let (mac, view_sum_c) =
                    view.buffer
                        .zip_fold(&weights.buffer[0], (0i32, 0i32), |acc, v, w| {
                            let v_c = i32::from_subset(&v.get(c).copied().unwrap_or(v[0]));
                            (acc.0 + v_c * i32::from_subset(&w[c]), acc.1 + v_c)
                        });
                let x = (mac, view_sum_c * weights_zero_point);
                // Per-channel sum of the weights overlapping the (unpadded) input
                // region; interior views reuse the precomputed sum, border views
                // subtract the weights landing on padding.
                let weights_sum_c = if view.len == WEIGHTS_ROWS * WEIGHTS_COLS {
                    constants.3[c]
                } else {
                    constants.3[c]
                        - weights.buffer[0].zip_fold(&view.mask, 0i32, |acc, w, m| {
                            if m {
                                acc
                            } else {
                                acc + i32::from_subset(&w[c])
                            }
                        })
                };
                // Corrected integer accumulator: dot product with the input and weight
                // zero points folded out, plus the integer bias.
                let acc = x.0 - x.1 - input_zero_point * weights_sum_c
                    + view.len as i32 * input_zero_point * weights_zero_point
                    + constants.0[c];
                // Requantize entirely in integer arithmetic, then add the output zero
                // point and clamp to the quantized range.
                let y = saturating_cast::<T>(
                    multiply_by_quantized_multiplier(
                        acc,
                        constants.1.get(c).copied().unwrap_or(constants.1[0]),
                        constants.2.get(c).copied().unwrap_or(constants.2[0]),
                    ) + output_zero_point_i32,
                );
                // Apply the fused activation function (if any)
                match options.fused_activation {
                    FusedActivation::None => y,
                    FusedActivation::Relu => relu(y, output_zero_point[0]),
                    FusedActivation::Relu6 => relu6(y, output_zero_point[0], relu6_max),
                }
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::matrix;

    use crate::tensor::Tensor2D;

    use super::*;

    const INPUT: Tensor4D<i8, 1, 2, 3, 2, 1> = Tensor4D {
        buffer: [matrix![
            [1, 2], [3, 4],  [5,  6];
            [7, 8], [9, 10], [11, 12]
        ]],
        scale: [0.13],
        zero_point: [14],
    };
    const WEIGHTS: Tensor4D<i8, 1, 2, 3, 2, 2> = Tensor4D {
        buffer: [matrix![
            [15, 16], [17, 18], [19, 20];
            [21, 22], [23, 24], [25, 26]
        ]],
        scale: [0.27, 0.28],
        zero_point: [29, 30],
    };
    const _BIASES: Tensor2D<i32, 2, 1, 2> = Tensor2D {
        buffer: matrix![
            31;
            32
        ],
        scale: [0.33, 0.34],
        zero_point: [35, 36],
    };
    const OUTPUT_SCALE: [f32; 1] = [0.37];
    const OUTPUT_ZERO_POINT: [i8; 1] = [38];
    const OPTIONS: DepthwiseConv2DOptions = DepthwiseConv2DOptions {
        fused_activation: FusedActivation::None,
        view_padding: TensorViewPadding::Same,
        strides: (1, 1),
    };
    const CONSTANTS: (
        Buffer2D<i32, 2, 1>,
        Buffer2D<i32, 2, 1>,
        Buffer2D<i32, 2, 1>,
        Buffer2D<i32, 2, 1>,
    ) = (
        matrix![-38; -37],
        matrix![1_629_765_969; 1_690_127_671],
        matrix![-3; -3],
        matrix![120; 126],
    );
    const OUTPUT: Tensor4D<i8, 1, 2, 3, 2, 1> = Tensor4D {
        buffer: [matrix![
            [66, 64], [82, 78], [65, 62];
            [47, 45], [52, 49], [45, 42]
        ]],
        scale: [0.37],
        zero_point: [38],
    };

    #[test]
    fn depthwise_conv_2d_layer() {
        assert_eq!(
            depthwise_conv_2d(
                INPUT,
                &WEIGHTS,
                OUTPUT_SCALE,
                OUTPUT_ZERO_POINT,
                OPTIONS,
                CONSTANTS,
            ),
            OUTPUT
        );
    }
}
