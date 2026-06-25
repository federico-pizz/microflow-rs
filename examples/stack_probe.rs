//! Host-side stack-usage probe for the nano_u chain.
//!
//! microflow's `predict_inner` is one sequential chain of `let input = op(input)`
//! shadows. In principle only two tensors (current input + layer output) are live
//! at once, so peak stack should be ~2x the largest tensor. In practice the whole
//! chain is `#[inline(always)]` into one giant frame and LLVM does not reuse the
//! dead slots, so peak ~= sum of all tensors (the "stack wall").
//!
//! This binary-searches the smallest thread stack on which `predict` succeeds,
//! giving a host proxy for that peak. Absolute bytes differ from Xtensa, but the
//! relative change between code variants is representative of whether slot reuse
//! kicks in.
//!
//! A thread stack overflow aborts the whole process (it cannot be caught by
//! `join`), so each candidate size is probed in a fresh subprocess: the parent
//! binary-searches and re-execs itself with `<kib>` as argv[1]; the child runs
//! `predict` on a thread of that size and exits 0 on success / aborts on overflow.
//!
//! Run with: `cargo run --release --example stack_probe`.

use microflow::buffer::Buffer2D;
use microflow_macros::model;
use std::process::Command;
use std::thread;

#[model("models/nano_u.tflite")]
struct NanoU;

fn child_run(stack_kib: usize) {
    thread::Builder::new()
        .stack_size(stack_kib * 1024)
        .spawn(|| {
            let input = [Buffer2D::from_element([0.0f32, 0.0, 0.0])];
            let output = NanoU::predict(input);
            std::hint::black_box(&output);
        })
        .unwrap()
        .join()
        .unwrap();
}

/// Spawn a fresh copy of this binary that probes `stack_kib`; return true if it
/// completed without overflowing (exit status success).
fn probe(stack_kib: usize) -> bool {
    let exe = std::env::current_exe().unwrap();
    Command::new(exe)
        .arg(stack_kib.to_string())
        .status()
        .unwrap()
        .success()
}

fn main() {
    if let Some(arg) = std::env::args().nth(1) {
        // Child mode: run one probe at the requested size.
        child_run(arg.parse().unwrap());
        return;
    }

    // Parent mode: binary-search the smallest stack (KiB) that does not overflow.
    let mut lo = 8usize; // known-too-small lower bound
    let mut hi = 8192usize; // generous upper bound
    assert!(probe(hi), "even {hi} KiB overflowed");
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if probe(mid) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    println!("\nnano_u predict minimum thread stack: ~{hi} KiB");
}
