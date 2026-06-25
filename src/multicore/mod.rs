//! Dual-core (ESP32-S3 APP-core) work dispatch for data-parallel inference.
//!
//! Enabled by the `multicore` feature. The model's `predict()` is a sequential
//! layer chain, but each heavy layer's output is embarrassingly parallel over
//! output pixels, so we split a layer's **output rows** across the two cores and
//! join with a barrier before the next layer (see `docs/ESP32_S3_MULTICORE.md`).
//!
//! ## Design (no heap, no RTOS, load/store atomics only)
//! The firmware starts the APP core (core 1) into [`worker_loop`]; core 0 runs
//! `predict()` as usual. When a heavy op calls [`parallel_for_rows`], core 0
//! publishes a job (a monomorphized trampoline + a pointer to the on-stack
//! closure + the row range), runs the *first* half itself, and waits for core 1
//! to finish the *second* half.
//!
//! Synchronization uses **single-writer generation counters** — `DISPATCH` is
//! written only by core 0, `COMPLETE` only by core 1 — so plain atomic
//! load/store suffice (no compare-and-swap, hence no `portable-atomic`
//! dependency). The `Release`/`Acquire` pair on these counters carries the
//! happens-before edge that makes the job fields and the cores' disjoint writes
//! visible across the boundary. On the ESP32-S3 internal SRAM is directly shared
//! (no per-core data cache), so no cache flush is needed — only this ordering.
//!
//! ## Safety
//! Three small, audited `unsafe` points, all justified here:
//! 1. [`SharedRows`] is `Send + Sync` around a `*mut`: sound because the two
//!    cores write **provably disjoint** row ranges and never form an aliasing
//!    `&mut`, and the barrier orders all writes before core 0 reads the result.
//! 2. The trampoline transmutes the stored `usize` back to a fn pointer.
//! 3. The job payload points at core 0's stack closure, which stays alive for
//!    the whole parallel region (core 0 blocks at the join), so core 1's
//!    dereference is valid.

use core::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};

/// Set true by [`worker_loop`] once the APP core is polling for jobs. Until then
/// (and on single-core builds) [`parallel_for_rows`] runs serially.
static WORKER_READY: AtomicBool = AtomicBool::new(false);

/// Job generation, written **only by core 0** to publish a job.
static DISPATCH: AtomicU32 = AtomicU32::new(0);
/// Completion generation, written **only by core 1** when it finishes a job.
static COMPLETE: AtomicU32 = AtomicU32::new(0);

/// Job fields, published (Relaxed) before the `Release` store of `DISPATCH`.
static JOB_RUN: AtomicUsize = AtomicUsize::new(0); // trampoline fn ptr as usize
static JOB_PAYLOAD: AtomicUsize = AtomicUsize::new(0); // *const closure as usize
static JOB_R0: AtomicUsize = AtomicUsize::new(0);
static JOB_R1: AtomicUsize = AtomicUsize::new(0);

/// A raw pointer to a tensor's element storage, marked shareable across cores.
///
/// # Safety
/// Callers must only ever write **disjoint** element indices from each core and
/// must not form an aliasing `&mut` to the pointee. The join barrier in
/// [`parallel_for_rows`] orders all writes before the result is read. Under
/// those conditions sharing the pointer is data-race free.
#[derive(Clone, Copy)]
pub struct SharedRows<T>(pub *mut T);
// SAFETY: see the type's contract — disjoint writes + barrier ordering.
unsafe impl<T> Send for SharedRows<T> {}
unsafe impl<T> Sync for SharedRows<T> {}

/// Whether the APP core is up and polling for jobs. The firmware can spin on
/// this after `start_app_core` so the first layer is already parallel.
#[inline]
pub fn worker_ready() -> bool {
    WORKER_READY.load(Ordering::Acquire)
}

/// The APP-core (core 1) entry point. The firmware starts the second core into
/// this function; it never returns. It waits for jobs published by core 0,
/// runs the assigned row band, and signals completion.
pub fn worker_loop() -> ! {
    WORKER_READY.store(true, Ordering::Release);
    let mut seen: u32 = 0;
    loop {
        let generation = DISPATCH.load(Ordering::Acquire);
        if generation == seen {
            core::hint::spin_loop();
            continue;
        }
        seen = generation;
        // SAFETY: published before the Release store of DISPATCH we just
        // Acquire-loaded; the value is a `tramp::<F>` fn pointer stored as usize.
        let run: unsafe fn(*const (), usize, usize) =
            unsafe { core::mem::transmute(JOB_RUN.load(Ordering::Relaxed)) };
        let payload = JOB_PAYLOAD.load(Ordering::Relaxed) as *const ();
        let r0 = JOB_R0.load(Ordering::Relaxed);
        let r1 = JOB_R1.load(Ordering::Relaxed);
        // SAFETY: `payload` points at the closure on core 0's stack, kept alive
        // until core 0 observes our COMPLETE store below.
        unsafe { run(payload, r0, r1) };
        COMPLETE.store(seen, Ordering::Release);
    }
}

/// Run `f(0, mid)` on core 0 and `f(mid, total_rows)` on core 1 in parallel,
/// returning once both halves are done. `f` must compute a half-open row band;
/// the two bands are disjoint, so a closure that writes per-row output through a
/// [`SharedRows`] pointer is data-race free.
///
/// Falls back to a single serial `f(0, total_rows)` when the worker is not
/// running (e.g. host builds/tests) or the work is too small to split.
#[inline]
pub fn parallel_for_rows<F>(total_rows: usize, f: F)
where
    F: Fn(usize, usize) + Sync,
{
    if total_rows < 2 || !WORKER_READY.load(Ordering::Acquire) {
        f(0, total_rows);
        return;
    }

    /// Monomorphized trampoline: reconstruct the closure ref and invoke it.
    unsafe fn tramp<F: Fn(usize, usize) + Sync>(p: *const (), r0: usize, r1: usize) {
        // SAFETY: `p` is `&f as *const F` published by `parallel_for_rows`, valid
        // for the whole parallel region (the caller blocks at the join).
        let f = unsafe { &*(p as *const F) };
        f(r0, r1);
    }

    let mid = total_rows / 2;
    JOB_RUN.store(tramp::<F> as usize, Ordering::Relaxed);
    JOB_PAYLOAD.store(&f as *const F as usize, Ordering::Relaxed);
    JOB_R0.store(mid, Ordering::Relaxed);
    JOB_R1.store(total_rows, Ordering::Relaxed);

    // Publish the job to core 1 (Release pairs with the worker's Acquire).
    let generation = DISPATCH.load(Ordering::Relaxed).wrapping_add(1);
    DISPATCH.store(generation, Ordering::Release);

    // Core 0 computes the first half while core 1 handles the second.
    f(0, mid);

    // Join: wait for core 1, with Acquire to observe its writes.
    while COMPLETE.load(Ordering::Acquire) != generation {
        core::hint::spin_loop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::AtomicI32;

    #[test]
    fn serial_fallback_covers_full_range() {
        // Worker never started -> parallel_for_rows must run f(0, total) once.
        let sum = AtomicI32::new(0);
        parallel_for_rows(10, |r0, r1| {
            for i in r0..r1 {
                sum.fetch_add(i as i32, Ordering::Relaxed);
            }
        });
        assert_eq!(sum.load(Ordering::Relaxed), 45); // 0+1+..+9
    }

    #[test]
    fn serial_fallback_handles_tiny_and_empty() {
        let calls = AtomicI32::new(0);
        for total in [0usize, 1] {
            parallel_for_rows(total, |r0, r1| {
                calls.fetch_add((r1 - r0) as i32, Ordering::Relaxed);
            });
        }
        // 0 rows + 1 row = 1 unit of work total.
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }
}
