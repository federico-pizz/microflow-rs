use microflow::buffer::Buffer2D;
use microflow_macros::model;

#[model("models/nano_u.tflite")]
struct NanoU;

/// FNV-1a 64-bit hash over the debug representation of the output tensor.
/// `nano_u` produces a large segmentation map, so instead of checking in a
/// multi-kilobyte fixture we pin a fingerprint of the full output. A latency
/// optimization that intentionally shifts quantized outputs (e.g. integer
/// multiply-shift requantization) will change this value — regenerate the
/// constant below and cross-check the small exact oracles (`person_detect`,
/// `sine`) which must stay bit-identical to TFLite.
fn fingerprint(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[test]
fn nano_u_model() {
    let input = [Buffer2D::from_element([0.0f32, 0.0, 0.0])];
    let output = NanoU::predict(input);
    let fp = fingerprint(&format!("{output:?}"));
    assert_eq!(
        fp, 0xa9f4_5525_9497_cee3,
        "nano_u output fingerprint changed: {fp:#018x}"
    );
}
