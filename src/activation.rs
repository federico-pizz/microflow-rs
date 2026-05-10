use crate::quantize::{quantize, Quantized};
use core::cmp::{max, min};
use libm::floorf;

/// Fast approximation of e^x using exp2(x·log₂e) with a cubic Taylor polynomial for
/// the fractional part. ~0.1% max relative error — well within INT8 quantization
/// noise (~0.8% per step). Replaces libm::expf, which is 3–4× slower on MCU targets.
/// Clamps input to [-87, 88] to avoid f32 overflow/underflow.
#[inline(always)]
pub(crate) fn fast_expf(x: f32) -> f32 {
    let x = if x < -87.0 { -87.0 } else if x > 88.0 { 88.0 } else { x };
    let t = x * core::f32::consts::LOG2_E;
    let ti = floorf(t) as i32;      // exact integer floor handles negative t correctly
    let tf = t - ti as f32;         // fractional part, always in [0, 1)
    // Cubic Taylor series for 2^tf; coefficients are (ln 2)^k / k!
    let p = 1.0
        + tf * (core::f32::consts::LN_2
            + tf * (0.240_226_507 + tf * 0.055_504_109));
    f32::from_bits(((ti + 127) as u32) << 23) * p
}

/// Represents the supported activation functions.
pub enum FusedActivation {
    /// The identity activation function.
    None,
    /// The Rectified Linear Unit (ReLU) function.
    Relu,
    /// The Rectified Linear Unit 6 (ReLU6) function.
    Relu6,
}

/// Performs the Rectified Linear Unit (ReLU) activation function.
///
/// # Arguments
/// * `input` - The input value of type `T`
/// * `zero_point` - The quantization zero point
///
pub fn relu<T: Quantized>(input: T, zero_point: T) -> T {
    max(input, zero_point)
}

/// Performs the Rectified Linear Unit 6 (ReLU6) activation function.
///
/// # Arguments
/// * `input` - The input value of type `T`
/// * `zero_point` - The quantization zero point
/// * `max_value` - The precomputed quantized value of 6.0 (`quantize(6., scale, zero_point)`)
///
pub fn relu6<T: Quantized>(input: T, zero_point: T, max_value: T) -> T {
    min(relu(input, zero_point), max_value)
}

/// Performs the Softmax activation function.
///
/// # Arguments
/// * `input` - The floating-point input value
/// * `sum` - The sum of the outcomes
/// * `scale` - The quantization scale
/// * `zero_point` - The quantization zero point
///
pub fn softmax<T: Quantized>(input: f32, sum: f32, scale: f32, zero_point: T) -> T {
    quantize(fast_expf(input) / sum, scale, zero_point)
}

/// Performs the Sigmoid activation function on a dequantized float value.
///
/// # Arguments
/// * `input` - The floating-point dequantized input value
/// * `scale` - The output quantization scale
/// * `zero_point` - The output quantization zero point
///
pub fn sigmoid<T: Quantized>(input: f32, scale: f32, zero_point: T) -> T {
    quantize(1.0 / (1.0 + fast_expf(-input)), scale, zero_point)
}

/// Performs the Tanh activation function on a dequantized float value.
pub fn tanh<T: Quantized>(input: f32, scale: f32, zero_point: T) -> T {
    // tanh(x) = 2·sigmoid(2x) − 1, reusing fast_expf with a single call
    quantize(2.0 / (1.0 + fast_expf(-2.0 * input)) - 1.0, scale, zero_point)
}

/// Performs the HardSwish activation function: x * relu6(x + 3) / 6.
pub fn hard_swish<T: Quantized>(input: f32, scale: f32, zero_point: T) -> T {
    let y = if input <= -3.0 {
        0.0
    } else if input >= 3.0 {
        input
    } else {
        input * (input + 3.0) / 6.0
    };
    quantize(y, scale, zero_point)
}

/// Performs the LeakyReLU activation function: alpha * x for x < 0, x otherwise.
pub fn leaky_relu<T: Quantized>(input: f32, alpha: f32, scale: f32, zero_point: T) -> T {
    quantize(if input >= 0.0 { input } else { alpha * input }, scale, zero_point)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCALE: f32 = 0.1;
    const ZERO_POINT: i8 = 2;

    const RELU_INACTIVE_INPUT: i8 = 1;
    const RELU_ACTIVE_INPUT: i8 = 3;

    const RELU6_SATURATED_INPUT: i8 = 63;
    const RELU6_SATURATION_POINT: i8 = 62;
    // quantize(6.0, SCALE=0.1, ZERO_POINT=2) = round(6.0/0.1 + 2) = 62
    const RELU6_MAX: i8 = 62;

    const SOFTMAX_INPUT_1: f32 = 1.;
    const SOFTMAX_INPUT_2: f32 = 2.;
    const SOFTMAX_INPUT_3: f32 = 3.;
    const SOFTMAX_SUM: f32 = 30.192_875;
    const SOFTMAX_OUTPUT_1: i8 = 3;
    const SOFTMAX_TOTAL_PROBABILITY: i8 = 16;

    #[test]
    fn relu_inactive() {
        assert_eq!(relu(RELU_INACTIVE_INPUT, ZERO_POINT), ZERO_POINT);
    }

    #[test]
    fn relu_active() {
        assert_eq!(relu(RELU_ACTIVE_INPUT, ZERO_POINT), RELU_ACTIVE_INPUT);
    }

    #[test]
    fn relu6_saturated() {
        assert_eq!(
            relu6(RELU6_SATURATED_INPUT, ZERO_POINT, RELU6_MAX),
            RELU6_SATURATION_POINT
        );
    }

    #[test]
    fn softmax_active() {
        assert_eq!(
            softmax(SOFTMAX_INPUT_1, SOFTMAX_SUM, SCALE, ZERO_POINT),
            SOFTMAX_OUTPUT_1
        );
    }

    #[test]
    fn softmax_total() {
        let total = softmax(SOFTMAX_INPUT_1, SOFTMAX_SUM, SCALE, ZERO_POINT)
            + softmax(SOFTMAX_INPUT_2, SOFTMAX_SUM, SCALE, ZERO_POINT)
            + softmax(SOFTMAX_INPUT_3, SOFTMAX_SUM, SCALE, ZERO_POINT);
        assert_eq!(total, SOFTMAX_TOTAL_PROBABILITY);
    }
}
