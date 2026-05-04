use core::array;

use crate::buffer::Buffer2D;
use crate::quantize::Quantized;
use crate::tensor::Tensor4D;

/// Splits a 4D tensor along the channel axis (axis=3) into two tensors.
///
/// The first `CHANS1` channels become `output1`; the remaining `CHANS2` channels become `output2`.
/// Both outputs inherit the input's scale and zero_point unchanged (no requantization).
/// `CHANS` must equal `CHANS1 + CHANS2` — verified only at runtime by buffer bounds.
#[inline(always)]
pub fn split_2<
    T: Quantized,
    const BATCHES: usize,
    const ROWS: usize,
    const COLS: usize,
    const CHANS: usize,
    const CHANS1: usize,
    const CHANS2: usize,
>(
    input: Tensor4D<T, BATCHES, ROWS, COLS, CHANS, 1>,
) -> (
    Tensor4D<T, BATCHES, ROWS, COLS, CHANS1, 1>,
    Tensor4D<T, BATCHES, ROWS, COLS, CHANS2, 1>,
)
where
    T: Copy,
{
    let scale = input.scale;
    let zero_point = input.zero_point;

    let out1 = array::from_fn(|b| {
        Buffer2D::from_fn(|i, j| array::from_fn(|c| input.buffer[b][(i, j)][c]))
    });
    let out2 = array::from_fn(|b| {
        Buffer2D::from_fn(|i, j| array::from_fn(|c| input.buffer[b][(i, j)][c + CHANS1]))
    });

    (
        Tensor4D::new(out1, scale, zero_point),
        Tensor4D::new(out2, scale, zero_point),
    )
}

#[cfg(test)]
mod tests {
    use nalgebra::matrix;

    use super::*;

    // input: 1×2×2×4, split into [1×2×2×2, 1×2×2×2]
    const INPUT: Tensor4D<i8, 1, 2, 2, 4, 1> = Tensor4D {
        buffer: [matrix![
            [1, 2, 10, 20], [3, 4, 30, 40];
            [5, 6, 50, 60], [7, 8, 70, 80]
        ]],
        scale: [0.1],
        zero_point: [0],
    };
    const OUT1: Tensor4D<i8, 1, 2, 2, 2, 1> = Tensor4D {
        buffer: [matrix![
            [1, 2], [3, 4];
            [5, 6], [7, 8]
        ]],
        scale: [0.1],
        zero_point: [0],
    };
    const OUT2: Tensor4D<i8, 1, 2, 2, 2, 1> = Tensor4D {
        buffer: [matrix![
            [10, 20], [30, 40];
            [50, 60], [70, 80]
        ]],
        scale: [0.1],
        zero_point: [0],
    };

    #[test]
    fn split_2_layer() {
        let (o1, o2) = split_2::<i8, 1, 2, 2, 4, 2, 2>(INPUT);
        assert_eq!(o1, OUT1);
        assert_eq!(o2, OUT2);
    }
}
