use libm::roundf;

use simba::scalar::SupersetOf;

use crate::activation::{relu, relu6, FusedActivation};
use crate::quantize::Quantized;
use crate::tensor::Tensor4D;

pub struct MulOptions {
    pub fused_activation: FusedActivation,
}

/// Performs element-wise multiplication of two quantized 4D tensors.
/// Returns a 4-dimensional output tensor containing the result of the operation.
///
/// # Arguments
/// * `input1` - The first 4-dimensional input tensor
/// * `input2` - The second 4-dimensional input tensor
/// * `output_scale` - The scale of the resulting output tensor
/// * `output_zero_point` - The zero point of the resulting output tensor
/// * `options` - Operator's options as a [`MulOptions`] struct
/// * `constants` - Pre-computed values:
///   `(c0, d1, d2, c2)` where `c0 = s1*s2/s_out`, `d1 = c0*zp1`, `d2 = c0*zp2`,
///   `c2 = f32(zp_out) + c0*zp1*zp2`
///
/// Formula: `output = round(c0*a*b - d1*b - d2*a + c2)`
///
#[inline(always)]
pub fn mul<
    T: Quantized,
    const BATCHES: usize,
    const ROWS: usize,
    const COLS: usize,
    const CHANS: usize,
>(
    mut input1: Tensor4D<T, BATCHES, ROWS, COLS, CHANS, 1>,
    input2: &Tensor4D<T, BATCHES, ROWS, COLS, CHANS, 1>,
    output_scale: [f32; 1],
    output_zero_point: [T; 1],
    options: MulOptions,
    constants: (f32, f32, f32, f32),
) -> Tensor4D<T, BATCHES, ROWS, COLS, CHANS, 1> {
    for b in 0..BATCHES {
        for i in 0..ROWS {
            for j in 0..COLS {
                // Copy both cells before writing to avoid aliasing
                let a_vals = input1.buffer[b][(i, j)];
                let b_vals = input2.buffer[b][(i, j)];
                for c in 0..CHANS {
                    let a = f32::from_subset(&a_vals[c]);
                    let bv = f32::from_subset(&b_vals[c]);
                    // Expand (a - zp1)(b - zp2) with precomputed coefficients to avoid
                    // redundant dequantize/requantize steps.
                    let y = T::from_superset_unchecked(&roundf(
                        constants.0 * a * bv - constants.1 * bv - constants.2 * a + constants.3,
                    ));
                    input1.buffer[b][(i, j)][c] = match options.fused_activation {
                        FusedActivation::None => y,
                        FusedActivation::Relu => relu(y, output_zero_point[0]),
                        FusedActivation::Relu6 => relu6(y, output_scale[0], output_zero_point[0]),
                    };
                }
            }
        }
    }
    input1.scale = output_scale;
    input1.zero_point = output_zero_point;
    input1
}

#[cfg(test)]
mod tests {
    use nalgebra::matrix;

    use super::*;

    // input1: scale=0.1, zero_point=0
    // input2: scale=0.1, zero_point=0
    // output: scale=0.01, zero_point=0
    // c0 = 0.1*0.1/0.01 = 1.0, d1=0, d2=0, c2=0
    const INPUT1: Tensor4D<i8, 1, 2, 2, 2, 1> = Tensor4D {
        buffer: [matrix![
            [2, 3], [4, 5];
            [1, 2], [3, 4]
        ]],
        scale: [0.1],
        zero_point: [0],
    };
    const INPUT2: Tensor4D<i8, 1, 2, 2, 2, 1> = Tensor4D {
        buffer: [matrix![
            [2, 2], [3, 3];
            [4, 4], [5, 5]
        ]],
        scale: [0.1],
        zero_point: [0],
    };
    const OUTPUT_SCALE: [f32; 1] = [0.01];
    const OUTPUT_ZERO_POINT: [i8; 1] = [0];
    const OPTIONS: MulOptions = MulOptions {
        fused_activation: FusedActivation::None,
    };
    // c0=1.0, d1=0.0, d2=0.0, c2=0.0 → output = round(a*b)
    const CONSTANTS: (f32, f32, f32, f32) = (1.0, 0.0, 0.0, 0.0);
    const OUTPUT: Tensor4D<i8, 1, 2, 2, 2, 1> = Tensor4D {
        buffer: [matrix![
            [4,  6],  [12, 15];
            [4,  8],  [15, 20]
        ]],
        scale: [0.01],
        zero_point: [0],
    };

    #[test]
    fn mul_layer() {
        assert_eq!(
            mul(INPUT1, &INPUT2, OUTPUT_SCALE, OUTPUT_ZERO_POINT, OPTIONS, CONSTANTS),
            OUTPUT
        );
    }
}
