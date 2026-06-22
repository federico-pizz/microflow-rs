use byterepr::ByteReprNum;
use nalgebra::Scalar;
use quote::ToTokens;
use simba::scalar::SubsetOf;

/// Represents the trait to constrain a type to be quantized and tokenized.
pub(crate) trait TokenQuantized:
    Scalar + ByteReprNum + ToTokens + SubsetOf<i32> + SubsetOf<f32> + SubsetOf<i64>
{
}

impl<T: Scalar + ByteReprNum + ToTokens + SubsetOf<i32> + SubsetOf<f32> + SubsetOf<i64>>
    TokenQuantized for T
{
}

/// Decomposes a floating-point requantization scale into the fixed-point
/// `(multiplier, shift)` pair consumed at runtime by
/// `microflow::quantize::multiply_by_quantized_multiplier`.
///
/// This is TFLite's `QuantizeMultiplier`: it writes `m = q * 2^shift` with
/// `q` in `[0.5, 1)` (a `frexp` decomposition) and returns
/// `multiplier = round(q * 2^31)` together with `shift`. The computation is
/// performed in `f64` so the compile-time constants are not degraded by `f32`
/// rounding. A zero scale maps to `(0, 0)`.
pub(crate) fn quantize_multiplier(scale: f64) -> (i32, i32) {
    if scale == 0.0 {
        return (0, 0);
    }
    // frexp: scale = mantissa * 2^shift, with mantissa in [0.5, 1).
    let mut shift = 0i32;
    let mut mantissa = scale;
    while mantissa >= 1.0 {
        mantissa /= 2.0;
        shift += 1;
    }
    while mantissa < 0.5 {
        mantissa *= 2.0;
        shift -= 1;
    }
    let mut quantized = (mantissa * (1i64 << 31) as f64).round() as i64;
    // Rounding can push the mantissa up to exactly 1.0; renormalize.
    if quantized == 1i64 << 31 {
        quantized /= 2;
        shift += 1;
    }
    // Scales too small to represent collapse to zero (matches TFLite).
    if shift < -31 {
        return (0, 0);
    }
    (quantized as i32, shift)
}

#[cfg(test)]
mod tests {
    use super::quantize_multiplier;

    #[test]
    fn quantize_multiplier_reference() {
        const HALF: i32 = 1 << 30;
        assert_eq!(quantize_multiplier(1.0), (HALF, 1));
        assert_eq!(quantize_multiplier(0.5), (HALF, 0));
        assert_eq!(quantize_multiplier(0.25), (HALF, -1));
        assert_eq!(quantize_multiplier(0.0), (0, 0));
    }
}
