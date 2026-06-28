//! Per-layer latency profiling (enabled by the `profiling` feature).
//!
//! microflow is `no_std` and platform-agnostic, so it cannot read a clock
//! itself. Instead the host/firmware installs a timestamp hook via [`set_now`]
//! (e.g. reading the ESP32 timer in microseconds). Each in-place op then brackets
//! its work with [`now`] and [`record`], appending `(op_type, duration)` to a
//! fixed table. After an inference, the caller reads the table back with
//! [`count`]/[`entry`] and prints a per-layer breakdown, then [`reset`]s.
//!
//! This is a diagnostic: it does not change results, and when the hook is unset
//! (or the feature is off) it is inert.

use core::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

/// Op-type tags recorded alongside each layer's duration.
pub const OP_CONV: u32 = 1;
pub const OP_DEPTHWISE: u32 = 2;
pub const OP_MAXPOOL: u32 = 3;
pub const OP_RESIZE: u32 = 4;

/// Maximum number of layers recorded per inference.
pub const CAP: usize = 64;

/// Installed timestamp hook: a `fn() -> u64` (microseconds) stored as `usize`.
static NOW_HOOK: AtomicUsize = AtomicUsize::new(0);

static IDX: AtomicUsize = AtomicUsize::new(0);
static DURATION: [AtomicU32; CAP] = [const { AtomicU32::new(0) }; CAP];
static OPTYPE: [AtomicU32; CAP] = [const { AtomicU32::new(0) }; CAP];

/// Install the timestamp source (microseconds). Pass a monotonic `fn() -> u64`.
#[inline]
pub fn set_now(f: fn() -> u64) {
    NOW_HOOK.store(f as usize, Ordering::Relaxed);
}

/// Read the current timestamp via the installed hook (0 if none installed).
#[inline]
pub fn now() -> u64 {
    let h = NOW_HOOK.load(Ordering::Relaxed);
    if h == 0 {
        return 0;
    }
    // SAFETY: `h` is a `fn() -> u64` pointer published by `set_now`.
    let f: fn() -> u64 = unsafe { core::mem::transmute(h) };
    f()
}

/// Append one layer's measurement: `op` is an `OP_*` tag, `start` came from
/// [`now`] at the op's entry.
#[inline]
pub fn record(op: u32, start: u64) {
    let dt = now().wrapping_sub(start);
    let i = IDX.fetch_add(1, Ordering::Relaxed);
    if i < CAP {
        DURATION[i].store(dt as u32, Ordering::Relaxed);
        OPTYPE[i].store(op, Ordering::Relaxed);
    }
}

/// Reset the table before an inference.
#[inline]
pub fn reset() {
    IDX.store(0, Ordering::Relaxed);
}

/// Number of layers recorded since the last [`reset`].
#[inline]
pub fn count() -> usize {
    IDX.load(Ordering::Relaxed).min(CAP)
}

/// The `(op_type, duration_us)` recorded for layer `i`.
#[inline]
pub fn entry(i: usize) -> (u32, u32) {
    (
        OPTYPE[i].load(Ordering::Relaxed),
        DURATION[i].load(Ordering::Relaxed),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A test-controlled fake clock so durations are deterministic.
    static CLOCK: AtomicU64Shim = AtomicU64Shim::new(0);

    // `core` has `AtomicU64` on host; alias to keep the use local to the test.
    use core::sync::atomic::AtomicU64 as AtomicU64Shim;

    fn fake_now() -> u64 {
        CLOCK.load(Ordering::Relaxed)
    }

    /// Single test function: the profile table is process-global state, so we keep
    /// every assertion in one sequential test to avoid cross-test races on the
    /// shared statics. Exercises the full hook -> record -> read-back contract,
    /// including duration arithmetic, op tags, `reset`, and the `CAP` ceiling.
    #[test]
    fn hook_record_readback_and_cap() {
        // Before any hook is installed, `now` must be inert (this test is the only
        // caller of `set_now` in the crate, so the hook is genuinely unset here).
        assert_eq!(now(), 0, "now() must be 0 until a hook is installed");

        set_now(fake_now);
        CLOCK.store(1_000, Ordering::Relaxed);
        assert_eq!(now(), 1_000, "now() must read through the installed hook");

        reset();
        assert_eq!(count(), 0, "reset() must clear the table");

        // Record two layers with known elapsed times.
        let start = now(); // 1_000
        CLOCK.store(1_042, Ordering::Relaxed); // +42 us
        record(OP_CONV, start);

        let start = now(); // 1_042
        CLOCK.store(1_042 + 7, Ordering::Relaxed); // +7 us
        record(OP_DEPTHWISE, start);

        assert_eq!(count(), 2);
        assert_eq!(entry(0), (OP_CONV, 42), "layer 0 op/duration mismatch");
        assert_eq!(entry(1), (OP_DEPTHWISE, 7), "layer 1 op/duration mismatch");

        // Recording past CAP must not panic and `count` must saturate at CAP, so a
        // model with more layers than the table can never index out of bounds.
        reset();
        let start = now();
        for _ in 0..(CAP + 5) {
            record(OP_MAXPOOL, start);
        }
        assert_eq!(count(), CAP, "count() must saturate at CAP, never exceed it");
    }
}
