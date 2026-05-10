use libm::roundf;

use simba::scalar::SupersetOf;

use crate::activation::{relu, relu6, FusedActivation};
use crate::quantize::{quantize, Quantized};
use crate::tensor::Tensor4D;

pub struct AddOptions {
    pub fused_activation: FusedActivation,
}

/// Performs element-wise addition of two quantized 4D tensors.
/// Returns a 4-dimensional output tensor containing the result of the operation.
///
/// # Arguments
/// * `input1` - The first 4-dimensional input tensor
/// * `input2` - The second 4-dimensional input tensor
/// * `output_scale` - The scale of the resulting output tensor
/// * `output_zero_point` - The zero point of the resulting output tensor
/// * `options` - Operator's options as an [`AddOptions`] struct
/// * `constants` - Pre-computed values `(s1/s_out, s2/s_out, zp_out - s1/s_out*zp1 - s2/s_out*zp2)`
///
#[inline(always)]
pub fn add<
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
    options: AddOptions,
    constants: (f32, f32, f32),
) -> Tensor4D<T, BATCHES, ROWS, COLS, CHANS, 1> {
    let relu6_max = quantize(6., output_scale[0], output_zero_point[0]);
    for b in 0..BATCHES {
        for i in 0..ROWS {
            for j in 0..COLS {
                // Copy both cells before writing to avoid aliasing
                let a_vals = input1.buffer[b][(i, j)];
                let b_vals = input2.buffer[b][(i, j)];
                for c in 0..CHANS {
                    let a = f32::from_subset(&a_vals[c]);
                    let bv = f32::from_subset(&b_vals[c]);
                    // constants = (s1/s_out, s2/s_out, zp_out - s1/s_out*zp1 - s2/s_out*zp2)
                    let y = T::from_superset_unchecked(&roundf(
                        constants.0 * a + constants.1 * bv + constants.2,
                    ));
                    input1.buffer[b][(i, j)][c] = match options.fused_activation {
                        FusedActivation::None => y,
                        FusedActivation::Relu => relu(y, output_zero_point[0]),
                        FusedActivation::Relu6 => relu6(y, output_zero_point[0], relu6_max),
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
    // input2: scale=0.2, zero_point=0
    // output: scale=0.1, zero_point=0
    // constants: (0.1/0.1, 0.2/0.1, 0 - 0 - 0) = (1.0, 2.0, 0.0)
    const INPUT1: Tensor4D<i8, 1, 2, 2, 2, 1> = Tensor4D {
        buffer: [matrix![
            [1, 2], [3, 4];
            [5, 6], [7, 8]
        ]],
        scale: [0.1],
        zero_point: [0],
    };
    const INPUT2: Tensor4D<i8, 1, 2, 2, 2, 1> = Tensor4D {
        buffer: [matrix![
            [1, 1], [2, 2];
            [3, 3], [4, 4]
        ]],
        scale: [0.2],
        zero_point: [0],
    };
    const OUTPUT_SCALE: [f32; 1] = [0.1];
    const OUTPUT_ZERO_POINT: [i8; 1] = [0];
    const OPTIONS: AddOptions = AddOptions {
        fused_activation: FusedActivation::None,
    };
    // c0=1.0, c1=2.0, c2=0.0
    // out[b][i,j][c] = round(1.0*a + 2.0*b + 0.0)
    // [0][0,0] = round(1*1+2*1, 1*2+2*1) = (3, 4)
    // [0][0,1] = round(1*3+2*2, 1*4+2*2) = (7, 8)
    // [0][1,0] = round(1*5+2*3, 1*6+2*3) = (11, 12)
    // [0][1,1] = round(1*7+2*4, 1*8+2*4) = (15, 16) → clamped to 127
    const CONSTANTS: (f32, f32, f32) = (1.0, 2.0, 0.0);
    const OUTPUT: Tensor4D<i8, 1, 2, 2, 2, 1> = Tensor4D {
        buffer: [matrix![
            [3,  4],  [7,  8];
            [11, 12], [15, 16]
        ]],
        scale: [0.1],
        zero_point: [0],
    };

    #[test]
    fn add_layer() {
        assert_eq!(
            add(INPUT1, &INPUT2, OUTPUT_SCALE, OUTPUT_ZERO_POINT, OPTIONS, CONSTANTS),
            OUTPUT
        );
    }
}
