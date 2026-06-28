//! Functional validation of the dual-core dispatch path on the host.
//!
//! On device, core 1 runs [`microflow::multicore::worker_loop`] and core 0 splits
//! each heavy layer's output rows across the two cores via `parallel_for_rows`.
//! Here we stand up a `std` thread as the "APP core" worker so the *real*
//! DISPATCH/COMPLETE barrier and the row-band split execute across two threads —
//! not the serial fallback. This catches band off-by-ones, missed/double-covered
//! rows, and any divergence of the two-core result from the serial golden.
//!
//! Run with the `multicore` feature (optionally `+buffer-reuse` to mirror the
//! device build):
//!   cargo test --features buffer-reuse,multicore --test multicore
//!
//! Without the feature the module compiles to nothing.
#![cfg(feature = "multicore")]

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;

use microflow::buffer::Buffer2D;
use microflow::multicore::{parallel_for_rows, worker_loop, worker_ready};
use microflow_macros::model;

#[model("models/nano_u.tflite")]
struct NanoU;

/// FNV-1a 64-bit hash, identical to `tests/nano_u.rs`, so the dual-core output can
/// be compared against the same pinned serial fingerprint.
fn fingerprint(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Everything runs in a single test: the dispatch protocol uses process-global
/// single-writer counters, so exactly one thread may play "core 0" at a time.
/// Starting one worker and driving all scenarios sequentially keeps that
/// invariant (parallel `#[test]` execution would otherwise contend the worker).
#[test]
fn dual_core_split_is_parallel_and_bit_identical() {
    thread::spawn(|| worker_loop());
    // Until the worker is actually polling, `parallel_for_rows` takes the serial
    // fallback — spin so the assertions below exercise the true two-core path.
    while !worker_ready() {
        std::hint::spin_loop();
    }

    // 1) Coverage invariant: every row in `0..N` is computed exactly once. An odd N
    //    forces an uneven mid split (8 vs 9), which is where an off-by-one in the
    //    band boundaries would show up as a gap or an overlap.
    const N: usize = 17;
    let hits: Vec<AtomicUsize> = (0..N).map(|_| AtomicUsize::new(0)).collect();
    let main_id = thread::current().id();
    let worker_ran = AtomicBool::new(false);

    parallel_for_rows(N, |r0, r1| {
        // 2) Real parallelism: one band must execute off the dispatcher thread.
        if thread::current().id() != main_id {
            worker_ran.store(true, Ordering::Relaxed);
        }
        for i in r0..r1 {
            hits[i].fetch_add(1, Ordering::Relaxed);
        }
    });

    for (i, h) in hits.iter().enumerate() {
        assert_eq!(
            h.load(Ordering::Relaxed),
            1,
            "row {i} was not computed exactly once (gap or overlap in the split)"
        );
    }
    assert!(
        worker_ran.load(Ordering::Relaxed),
        "second band never ran on the worker thread — split silently fell back to serial"
    );

    // 3) End-to-end: the whole nano_u model, every heavy layer split across the two
    //    threads, must be bit-identical to the serial golden in `tests/nano_u.rs`.
    //    Integer results are order-independent per output pixel, so a divergence
    //    here means the row split corrupted or skipped output — the core claim.
    let input = [Buffer2D::from_element([0.0f32, 0.0, 0.0])];
    let output = NanoU::predict(input);
    let fp = fingerprint(&format!("{output:?}"));
    assert_eq!(
        fp, 0x0e4d_491d_1e63_de08,
        "dual-core output diverged from the serial golden: {fp:#018x}"
    );
}
