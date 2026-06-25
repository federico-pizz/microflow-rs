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
