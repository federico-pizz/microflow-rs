use libm::roundf;
use nalgebra::Scalar;
use num_traits::Bounded;
use simba::scalar::{SubsetOf, SupersetOf};

/// Represents the trait to constrain a type to be quantized.
pub trait Quantized: Scalar + Copy + Ord + Bounded + SubsetOf<i32> + SubsetOf<f32> {}
impl<T: Scalar + Copy + Ord + Bounded + SubsetOf<i32> + SubsetOf<f32>> Quantized for T {}

/// Performs quantization on the given floating-point input.
///
/// # Arguments
/// * `input` - The input value to quantize
/// * `scale` - The quantization scale
/// * `zero_point` - The quantization zero point
///
pub fn quantize<T: Quantized>(input: f32, scale: f32, zero_point: T) -> T {
    roundf(input / scale + f32::from_subset(&zero_point)).to_subset_unchecked()
}

/// Performs dequantization on the given integer input.
///
/// # Arguments
/// * `input` - The input value to dequantize
/// * `scale` - The quantization scale
/// * `zero_point` - The quantization zero point
///
pub fn dequantize<T: Quantized>(input: T, scale: f32, zero_point: T) -> f32 {
    scale * (f32::from_subset(&input) - f32::from_subset(&zero_point))
}

/// Rounds an already-scaled floating-point accumulator to the nearest integer
/// (ties away from zero) and narrows it to the quantized type, saturating to the
/// representable range as TFLite does.
///
/// This is the requantization tail shared by every quantized kernel. It avoids
/// `libm::roundf`, which is a full software routine on targets without an FPU
/// (e.g. Cortex-M0/M3, AVR, Xtensa LX6): adding `±0.5` and letting the `f32 -> T`
/// cast truncate toward zero yields round-half-away-from-zero in a couple of
/// instructions. The cast saturates, preserving the int8/uint8 clamping that the
/// previous `roundf` + unchecked conversion relied on.
#[inline(always)]
pub fn requantize<T: Quantized>(value: f32) -> T {
    T::from_superset_unchecked(&(value + 0.5_f32.copysign(value)))
}

/// Clamps an `i32` accumulator to the representable range of the quantized type
/// and narrows it, using only integer arithmetic. This is the integer counterpart
/// of [`requantize`]'s saturating `f32 -> T` cast: it performs the int8/uint8
/// clamping that the integer requantization tail relies on without touching the
/// FPU, which matters on targets where float conversions are software routines.
#[inline(always)]
pub fn saturating_cast<T: Quantized>(value: i32) -> T {
    let min = i32::from_subset(&T::min_value());
    let max = i32::from_subset(&T::max_value());
    T::from_superset_unchecked(&value.clamp(min, max))
}

/// Computes `round(x * multiplier * 2^shift)` using only integer arithmetic,
/// where `multiplier` is a Q0.31 fixed-point fraction in `[2^30, 2^31)` and
/// `shift` is the accompanying power-of-two exponent (positive = left shift,
/// negative = right shift). This mirrors TFLite's `MultiplyByQuantizedMultiplier`
/// (gemmlowp fixed-point), so the integer requantization path matches the
/// reference quantized kernels bit-for-bit. The `(multiplier, shift)` pair is
/// produced at compile time from the float requantization scale by the macro's
/// `quantize_multiplier`.
#[inline(always)]
pub fn multiply_by_quantized_multiplier(x: i32, multiplier: i32, shift: i32) -> i32 {
    let left_shift = if shift > 0 { shift } else { 0 };
    let right_shift = if shift > 0 { 0 } else { -shift };
    rounding_divide_by_pot(
        saturating_rounding_doubling_high_mul(x * (1 << left_shift), multiplier),
        right_shift,
    )
}

/// Returns the high 32 bits of the doubled 64-bit product of two Q0.31 values,
/// with rounding and saturation (gemmlowp's `SaturatingRoundingDoublingHighMul`).
#[inline(always)]
fn saturating_rounding_doubling_high_mul(a: i32, b: i32) -> i32 {
    if a == i32::MIN && b == i32::MIN {
        return i32::MAX;
    }
    let ab = i64::from(a) * i64::from(b);
    let nudge: i64 = if ab >= 0 { 1 << 30 } else { 1 - (1 << 30) };
    ((ab + nudge) / (1i64 << 31)) as i32
}

/// Divides by `2^exponent` with round-half-up semantics (gemmlowp's
/// `RoundingDivideByPOT`).
#[inline(always)]
fn rounding_divide_by_pot(x: i32, exponent: i32) -> i32 {
    if exponent == 0 {
        return x;
    }
    let mask = (1i32 << exponent) - 1;
    let remainder = x & mask;
    let threshold = (mask >> 1) + i32::from(x < 0);
    (x >> exponent) + i32::from(remainder > threshold)
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALUE: f32 = 1.;
    const SCALE: f32 = 0.2;
    const ZERO_POINT: i8 = 3;
    const VALUE_QUANTIZED: i8 = 8;
    const VALUE_DEQUANTIZED: f32 = 1.;

    #[test]
    fn quantize_value() {
        assert_eq!(quantize(VALUE, SCALE, ZERO_POINT), VALUE_QUANTIZED);
    }

    #[test]
    fn dequantize_value() {
        assert_eq!(
            dequantize(VALUE_QUANTIZED, SCALE, ZERO_POINT),
            VALUE_DEQUANTIZED
        );
    }

    #[test]
    fn saturating_cast_clamps_to_range() {
        assert_eq!(saturating_cast::<i8>(5), 5);
        assert_eq!(saturating_cast::<i8>(200), 127);
        assert_eq!(saturating_cast::<i8>(-200), -128);
        assert_eq!(saturating_cast::<u8>(-5), 0);
        assert_eq!(saturating_cast::<u8>(300), 255);
    }

    #[test]
    fn multiply_by_quantized_multiplier_matches_reference() {
        // (multiplier, shift) pairs as produced by `quantize_multiplier`:
        //   M = 1.00 -> (2^30,  1)   M = 0.50 -> (2^30,  0)   M = 0.25 -> (2^30, -1)
        const HALF: i32 = 1 << 30;
        assert_eq!(multiply_by_quantized_multiplier(4, HALF, 1), 4); // 4 * 1.00
        assert_eq!(multiply_by_quantized_multiplier(3, HALF, 1), 3);
        assert_eq!(multiply_by_quantized_multiplier(4, HALF, 0), 2); // 4 * 0.50
        assert_eq!(multiply_by_quantized_multiplier(8, HALF, -1), 2); // 8 * 0.25
                                                                      // Round half up toward positive infinity.
        assert_eq!(multiply_by_quantized_multiplier(3, HALF, 0), 2); // 1.5 -> 2
        assert_eq!(multiply_by_quantized_multiplier(-3, HALF, 0), -1); // -1.5 -> -1
    }
}
