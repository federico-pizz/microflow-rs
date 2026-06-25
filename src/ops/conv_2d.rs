use core::array;

use simba::scalar::SupersetOf;

use crate::activation::{relu, relu6, FusedActivation};
use crate::buffer::Buffer2D;
use crate::quantize::{multiply_by_quantized_multiplier, quantize, saturating_cast, Quantized};
use crate::tensor::{Tensor4D, TensorView, TensorViewPadding};

pub struct Conv2DOptions {
    pub fused_activation: FusedActivation,
    pub view_padding: TensorViewPadding,
    pub strides: (usize, usize),
}

/// Performs the Conv2D operation.
/// Returns a 4-dimensional output tensor containing the result of the operation.
///
/// # Arguments
/// * `input` - The 4-dimensional input tensor
/// * `filters` - The 4-dimensional tensor representing the filters of the operator
/// * `output_scale` - The scale of the resulting output tensor
/// * `output_zero_point` - The zero point of the resulting output tensor
/// * `options` - Operator's options as an [`Conv2DOptions`] struct
/// * `constants` - Constant values coming from the pre-processing phase
///
#[inline(always)]
pub fn conv_2d<
    // `Sync` lets the per-pixel closure (which borrows only the immutable input,
    // filters and constants) be shared across the two cores under `multicore`.
    // All instantiated quantized types (`i8`, `u8`) are `Sync`, so this does not
    // restrict callers.
    T: Quantized + Sync,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const INPUT_CHANS: usize,
    const FILTERS_BATCHES: usize,
    const FILTERS_ROWS: usize,
    const FILTERS_COLS: usize,
    const FILTERS_QUANTS: usize,
    const OUTPUT_ROWS: usize,
    const OUTPUT_COLS: usize,
>(
    input: Tensor4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS, 1>,
    filters: &Tensor4D<T, FILTERS_BATCHES, FILTERS_ROWS, FILTERS_COLS, INPUT_CHANS, FILTERS_QUANTS>,
    output_scale: [f32; 1],
    output_zero_point: [T; 1],
    options: Conv2DOptions,
    constants: (
        Buffer2D<i32, FILTERS_BATCHES, 1>,
        Buffer2D<i32, FILTERS_QUANTS, 1>,
        Buffer2D<i32, FILTERS_QUANTS, 1>,
        Buffer2D<i32, FILTERS_BATCHES, 1>,
    ),
) -> Tensor4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, FILTERS_BATCHES, 1> {
    // Thin by-value wrapper around the in-place variant. Used by host unit tests
    // and by the non-`buffer-reuse` codegen path.
    let mut output = Tensor4D::new(
        [Buffer2D::from_element(
            [output_zero_point[0]; FILTERS_BATCHES],
        )],
        output_scale,
        output_zero_point,
    );
    conv_2d_into(
        &mut output,
        &input,
        filters,
        output_scale,
        output_zero_point,
        options,
        constants,
    );
    output
}

/// Performs the Conv2D operation in place, writing the result into a
/// caller-owned output tensor instead of returning a fresh one.
///
/// This is the buffer-reuse entry point: by filling `&mut output` (rather than
/// returning a value the caller must move), LLVM can coalesce the transient
/// per-pixel results directly into the destination buffer's stack slot, which is
/// what lets the generated ping-pong chain hit the ~2×MAX stack floor.
///
/// # Arguments
/// * `output` - The 4-dimensional output tensor to fill
/// * `input` - The 4-dimensional input tensor
/// * `filters` - The 4-dimensional tensor representing the filters of the operator
/// * `output_scale` - The scale of the resulting output tensor
/// * `output_zero_point` - The zero point of the resulting output tensor
/// * `options` - Operator's options as an [`Conv2DOptions`] struct
/// * `constants` - Constant values coming from the pre-processing phase
///
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn conv_2d_into<
    // `Sync` lets the per-pixel closure (which borrows only the immutable input,
    // filters and constants) be shared across the two cores under `multicore`.
    // All instantiated quantized types (`i8`, `u8`) are `Sync`, so this does not
    // restrict callers.
    T: Quantized + Sync,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const INPUT_CHANS: usize,
    const FILTERS_BATCHES: usize,
    const FILTERS_ROWS: usize,
    const FILTERS_COLS: usize,
    const FILTERS_QUANTS: usize,
    const OUTPUT_ROWS: usize,
    const OUTPUT_COLS: usize,
>(
    output: &mut Tensor4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, FILTERS_BATCHES, 1>,
    input: &Tensor4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS, 1>,
    filters: &Tensor4D<T, FILTERS_BATCHES, FILTERS_ROWS, FILTERS_COLS, INPUT_CHANS, FILTERS_QUANTS>,
    output_scale: [f32; 1],
    output_zero_point: [T; 1],
    options: Conv2DOptions,
    constants: (
        Buffer2D<i32, FILTERS_BATCHES, 1>,
        Buffer2D<i32, FILTERS_QUANTS, 1>,
        Buffer2D<i32, FILTERS_QUANTS, 1>,
        Buffer2D<i32, FILTERS_BATCHES, 1>,
    ),
) {
    output.scale = output_scale;
    output.zero_point = output_zero_point;

    // Per-output-pixel computation: the filter-batch vector at output `(i, j)`.
    // Shared by the single-core path and (under `multicore`) by both cores, each
    // filling a disjoint band of output rows. It borrows only shared, immutable
    // data, so it is `Sync` and safe to invoke concurrently.
    let pixel = |i: usize, j: usize| -> [T; FILTERS_BATCHES] {
        // Extract the view using the view extraction algorithm
        let view: TensorView<T, FILTERS_ROWS, FILTERS_COLS, INPUT_CHANS> =
            input.view((i, j), 0, options.view_padding, options.strides);
        // Compute the view element sum once; it is shared across all filter batches.
        let view_sum = view.buffer.fold(0i32, |acc, a| {
            acc + a.iter().fold(0i32, |acc, e| acc + i32::from_subset(e))
        });
        // Perform the convolution for each filter batch
        array::from_fn(|b| {
            let relu6_max = quantize(6., output_scale[0], output_zero_point[0]);
            let input_zero_point = i32::from_subset(&input.zero_point[0]);
            let output_zero_point_i32 = i32::from_subset(&output_zero_point[0]);
            let filters_zero_point = i32::from_subset(
                &filters
                    .zero_point
                    .get(b)
                    .copied()
                    .unwrap_or(filters.zero_point[0]),
            );
            let x = (
                // Perform the dot product between the input region and the filter
                view.buffer.zip_fold(&filters.buffer[b], 0i32, |acc, v, f| {
                    acc + v
                        .iter()
                        .zip(f.iter())
                        .map(|(e1, e2)| i32::from_subset(e1) * i32::from_subset(e2))
                        .sum::<i32>()
                }),
                // Use the hoisted view sum; only multiply by the per-batch filter zero point
                view_sum * filters_zero_point,
            );
            // Sum of the filter weights overlapping the (unpadded) input region.
            // For interior views the whole filter is in bounds, so the precomputed
            // per-batch sum is used directly; only border views pay for a fold,
            // subtracting the weights that fall on padding.
            let filter_sum = if view.len == FILTERS_ROWS * FILTERS_COLS {
                constants.3[b]
            } else {
                constants.3[b]
                    - filters.buffer[b].zip_fold(&view.mask, 0i32, |acc, f, m| {
                        if m {
                            acc
                        } else {
                            acc + f.iter().fold(0i32, |acc, e| acc + i32::from_subset(e))
                        }
                    })
            };
            // Corrected integer accumulator: the dot product with the input and
            // filter zero points folded out, plus the integer bias.
            let acc = x.0 - x.1 - input_zero_point * filter_sum
                + view.len as i32 * INPUT_CHANS as i32 * input_zero_point * filters_zero_point
                + constants.0[b];
            // Requantize entirely in integer arithmetic, then add the output zero
            // point and clamp to the quantized range.
            let y = saturating_cast::<T>(
                multiply_by_quantized_multiplier(
                    acc,
                    constants.1.get(b).copied().unwrap_or(constants.1[0]),
                    constants.2.get(b).copied().unwrap_or(constants.2[0]),
                ) + output_zero_point_i32,
            );
            // Apply the fused activation function (if any)
            match options.fused_activation {
                FusedActivation::None => y,
                FusedActivation::Relu => relu(y, output_zero_point[0]),
                FusedActivation::Relu6 => relu6(y, output_zero_point[0], relu6_max),
            }
        })
    };

    // Single-core: fill the caller-owned output buffer in place, row by row.
    #[cfg(not(feature = "multicore"))]
    {
        let buffer = &mut output.buffer[0];
        for j in 0..OUTPUT_COLS {
            for i in 0..OUTPUT_ROWS {
                buffer[(i, j)] = pixel(i, j);
            }
        }
    }

    // Dual-core: split the output rows across both cores, each writing its own
    // disjoint band directly into the (already caller-owned) output buffer via a
    // shared raw pointer. No extra output buffer is allocated — that is exactly
    // why this needs `buffer-reuse`: the destination already lives in a reused
    // ping-pong slot, so the second core's band costs no additional stack. The
    // join barrier in `parallel_for_rows` orders all writes before the result is
    // read.
    #[cfg(feature = "multicore")]
    {
        let base: crate::multicore::SharedRows<[T; FILTERS_BATCHES]> =
            crate::multicore::SharedRows(output.buffer[0].as_mut_slice().as_mut_ptr());
        crate::multicore::parallel_for_rows(OUTPUT_ROWS, |r0, r1| {
            // Capture the whole `SharedRows` (Copy, Sync), not the bare `*mut`.
            let base = base;
            for i in r0..r1 {
                for j in 0..OUTPUT_COLS {
                    // SAFETY: column-major index `j*OUTPUT_ROWS + i` is in bounds
                    // (i < OUTPUT_ROWS, j < OUTPUT_COLS); the two cores cover
                    // disjoint row bands, so writes never alias.
                    unsafe { base.0.add(j * OUTPUT_ROWS + i).write(pixel(i, j)) };
                }
            }
        });
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
    const FILTERS: Tensor4D<i8, 2, 2, 3, 2, 2> = Tensor4D {
        buffer: [
            matrix![
                [15, 16], [17, 18], [19, 20];
                [21, 22], [23, 24], [25, 26]
            ],
            matrix![
                [27, 28], [29, 30], [31, 32];
                [33, 34], [35, 36], [37, 38]
            ],
        ],
        scale: [0.39, 0.40],
        zero_point: [41, 42],
    };
    const _BIASES: Tensor2D<i32, 2, 1, 2> = Tensor2D {
        buffer: matrix![
            43;
            44
        ],
        scale: [0.45, 0.46],
        zero_point: [47, 48],
    };
    const OUTPUT_SCALE: [f32; 1] = [0.49];
    const OUTPUT_ZERO_POINT: [i8; 1] = [50];
    const OPTIONS: Conv2DOptions = Conv2DOptions {
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
        matrix![-36; -35],
        matrix![1_777_590_546; 1_823_169_791],
        matrix![-3; -3],
        matrix![246; 390],
    );
    const OUTPUT: Tensor4D<i8, 1, 2, 3, 2, 1> = Tensor4D {
        buffer: [matrix![
            [127, 116], [127, 127], [127, 113];
            [98,  74],  [114, 84],  [82,  67]
        ]],
        scale: [0.49],
        zero_point: [50],
    };

    #[test]
    fn conv_2d_layer() {
        assert_eq!(
            conv_2d(
                INPUT,
                &FILTERS,
                OUTPUT_SCALE,
                OUTPUT_ZERO_POINT,
                OPTIONS,
                CONSTANTS,
            ),
            OUTPUT
        );
    }
}
