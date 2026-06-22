use core::array;

use simba::scalar::SupersetOf;

use crate::activation::{relu, relu6, FusedActivation};
use crate::buffer::Buffer2D;
use crate::quantize::{multiply_by_quantized_multiplier, quantize, saturating_cast, Quantized};
use crate::tensor::Tensor4D;

pub struct TransposeConv2DOptions {
    pub fused_activation: FusedActivation,
    pub strides: (usize, usize),
}

/// Performs the TransposeConv2D (deconvolution / fractionally-strided convolution) operation.
/// Upsamples the spatial dimensions of the input by the stride factor.
/// Returns a 4-dimensional output tensor containing the result of the operation.
///
/// # Arguments
/// * `input` - The 4-dimensional input tensor with shape (1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS)
/// * `filters` - The 4-dimensional filter tensor with shape (OUTPUT_CHANS, KH, KW, INPUT_CHANS)
/// * `output_scale` - The scale of the resulting output tensor
/// * `output_zero_point` - The zero point of the resulting output tensor
/// * `options` - Operator's options as a [`TransposeConv2DOptions`] struct
/// * `constants` - Pre-computed values `(bias[OUTPUT_CHANS], mac_scale[FILTERS_QUANTS])`
///   matching the same convention as `conv_2d`.
///
/// For each output position (oh, ow), this gathers contributions from all input positions
/// (h, w) where `oh = h * stride_h + kh` and `ow = w * stride_w + kw` for some kernel
/// offset (kh, kw), applying the same quantization correction as Conv2D.
///
#[inline(always)]
pub fn transpose_conv_2d<
    T: Quantized,
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
    options: TransposeConv2DOptions,
    constants: (
        Buffer2D<i32, FILTERS_BATCHES, 1>,
        Buffer2D<i32, FILTERS_QUANTS, 1>,
        Buffer2D<i32, FILTERS_QUANTS, 1>,
    ),
) -> Tensor4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, FILTERS_BATCHES, 1> {
    let relu6_max = quantize(6., output_scale[0], output_zero_point[0]);
    let input_zp = i32::from_subset(&input.zero_point[0]);
    let output_zero_point_i32 = i32::from_subset(&output_zero_point[0]);
    let output = [Buffer2D::from_fn(|oh, ow| {
        array::from_fn(|co| {
            let filter_zp = i32::from_subset(
                &filters
                    .zero_point
                    .get(co)
                    .copied()
                    .unwrap_or(filters.zero_point[0]),
            );
            // Gather all input positions (h, w) and kernel offsets (kh, kw) that map to
            // this output position: oh = h * sh + kh, ow = w * sw + kw.
            let mut mac = 0i32;
            let mut input_sum = 0i32; // sum of raw input values (for filter_zp correction)
            let mut filter_sum = 0i32; // sum of raw filter values (for input_zp correction)
            let mut valid_count = 0i32;

            for kh in 0..FILTERS_ROWS {
                if oh >= kh && (oh - kh) % options.strides.0 == 0 {
                    let h = (oh - kh) / options.strides.0;
                    if h < INPUT_ROWS {
                        for kw in 0..FILTERS_COLS {
                            if ow >= kw && (ow - kw) % options.strides.1 == 0 {
                                let w = (ow - kw) / options.strides.1;
                                if w < INPUT_COLS {
                                    for ci in 0..INPUT_CHANS {
                                        let a = i32::from_subset(&input.buffer[0][(h, w)][ci]);
                                        let f = i32::from_subset(&filters.buffer[co][(kh, kw)][ci]);
                                        mac += a * f;
                                        input_sum += a;
                                        filter_sum += f;
                                    }
                                    valid_count += INPUT_CHANS as i32;
                                }
                            }
                        }
                    }
                }
            }

            // Apply the same quantization correction as Conv2D:
            // corrected = mac - filter_zp*input_sum - input_zp*filter_sum + valid*input_zp*filter_zp
            let corrected = mac - filter_zp * input_sum - input_zp * filter_sum
                + valid_count * input_zp * filter_zp;

            // Fold the integer bias into the corrected accumulator, then requantize
            // entirely in integer arithmetic.
            let acc = corrected + constants.0[co];
            let y = saturating_cast::<T>(
                multiply_by_quantized_multiplier(
                    acc,
                    constants.1.get(co).copied().unwrap_or(constants.1[0]),
                    constants.2.get(co).copied().unwrap_or(constants.2[0]),
                ) + output_zero_point_i32,
            );
            match options.fused_activation {
                FusedActivation::None => y,
                FusedActivation::Relu => relu(y, output_zero_point[0]),
                FusedActivation::Relu6 => relu6(y, output_zero_point[0], relu6_max),
            }
        })
    })];
    Tensor4D::new(output, output_scale, output_zero_point)
}

#[cfg(test)]
mod tests {
    use nalgebra::matrix;

    use crate::tensor::Tensor2D;

    use super::*;

    // input: (1, 1, 2, 1), scale=0.1, zero_point=0
    // filter: (2, 2, 1, 1), scale=0.1, zero_point=0 → 2 output channels, 2×1 kernel
    // stride: (2, 1) → output shape (1, 1*2+2-1, 2*1+1-1, 2) = (1, 3, 2, 2)
    //
    // For stride=(2,1) and no zero points:
    //   input = [[1, 2]]  (h=0, w=0..1)
    //   filter[0] = [[3], [4]]   filter[1] = [[5], [6]]
    //
    // output[oh, ow, co]:
    //   (oh=0,ow=0): kh=0→h=0, kw=0→w=0: a=1,f[0][0,0]=3 → mac=3; corrected=3
    //   (oh=0,ow=1): kh=0→h=0, kw=0→w=1: a=2,f=3 → mac=6; corrected=6
    //   (oh=1,ow=0): kh=1→h=0, kw=0→w=0: a=1,f[0][1,0]=4 → mac=4; corrected=4
    //   (oh=1,ow=1): kh=1→h=0, kw=0→w=1: a=2,f=4 → mac=8; corrected=8
    //   (oh=2,ow=0): no valid (h=(2-0)/2=1 ≥ INPUT_ROWS=1 invalid; kh=1→h=(2-1)/2=0 but (2-1)%2≠0)
    //                → mac=0, corrected=0
    //   (oh=2,ow=1): similarly 0
    //
    // output_scale=0.01, output_zero_point=0
    // constants.0 = [0,0] (no bias), constants.1 = [0.1*0.1/0.01=1.0, 1.0]
    // output co=0: oh=0→3, oh=1→4, oh=2→0
    // output co=1: oh=0→5,  oh=1→6, oh=2→0
    const INPUT: Tensor4D<i8, 1, 1, 2, 1, 1> = Tensor4D {
        buffer: [matrix![[1], [2]]],
        scale: [0.1],
        zero_point: [0],
    };
    const FILTERS: Tensor4D<i8, 2, 2, 1, 1, 2> = Tensor4D {
        buffer: [matrix![[3]; [4]], matrix![[5]; [6]]],
        scale: [0.1, 0.1],
        zero_point: [0, 0],
    };
    const _BIASES: Tensor2D<i32, 2, 1, 2> = Tensor2D {
        buffer: matrix![0; 0],
        scale: [0.01, 0.01],
        zero_point: [0, 0],
    };
    const OUTPUT_SCALE: [f32; 1] = [0.01];
    const OUTPUT_ZERO_POINT: [i8; 1] = [0];
    const OPTIONS: TransposeConv2DOptions = TransposeConv2DOptions {
        fused_activation: FusedActivation::None,
        strides: (2, 1),
    };
    // bias=0 for both channels, M = input_scale * filter_scale / output_scale = 0.1*0.1/0.01 = 1.0,
    // which decomposes to multiplier=2^30, shift=1.
    const CONSTANTS: (
        Buffer2D<i32, 2, 1>,
        Buffer2D<i32, 2, 1>,
        Buffer2D<i32, 2, 1>,
    ) = (
        matrix![0; 0],
        matrix![1_073_741_824; 1_073_741_824],
        matrix![1; 1],
    );
    const OUTPUT: Tensor4D<i8, 1, 3, 2, 2, 1> = Tensor4D {
        buffer: [matrix![
            [3, 5], [6, 10];
            [4, 6], [8, 12];
            [0, 0], [0,  0]
        ]],
        scale: [0.01],
        zero_point: [0],
    };

    #[test]
    fn transpose_conv_2d_layer() {
        assert_eq!(
            transpose_conv_2d(
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
