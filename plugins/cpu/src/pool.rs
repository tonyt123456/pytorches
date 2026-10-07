//! The thread pool every CPU kernel runs on, plus helpers for splitting work into tasks.
//!
//! `PYTORCHES_CPU_THREADS` overrides the thread count (default: all logical cores) and
//! `PYTORCHES_CPU_SPIN_US` how long idle workers keep spinning before they sleep (default 100).
//!
//! Workers *spin* for a short while after the last job instead of sleeping at once. Waking a sleeping
//! thread costs 50-120 us on Windows, which is longer than a small matmul takes, and a training loop
//! has Python running between ops; the same trick is why OpenMP-based libraries keep threads hot.
//! After the spin window workers park, so an idle process does not burn CPU.
//!
//! The window is short on purpose. Measured on a hybrid-core laptop (a training-step loop, 16 threads):
//! no spinning loses most of the small-op speed (MLP step 520-700 vs 1,060-1,180 steps/s), but a
//! window of 300 us or more makes large ops slower (matmul 1024: ~720 -> 380-520 GFLOP/s, add: 78 ->
//! 46 GB/s), likely because workers parked in a spin loop get moved to slow cores. 100 us covers the
//! gap Python leaves between ops.
//!
//! Work is cut into fixed-size chunks that do not depend on the thread count, so results are
//! bit-identical however many threads run them. The cores of a hybrid CPU differ in speed, so tasks
//! are small and claimed dynamically.

use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// The task function of the current job, with its lifetime erased. The caller of `run_tasks` does not
/// return until every task has finished, so the pointee outlives every use.
#[derive(Clone, Copy)]
struct Job {
    f: *const (dyn Fn(usize) + Sync),
    n: usize,
    epoch: u32,
}
unsafe impl Send for Job {}

struct Shared {
    threads: usize,
    spin: Duration,
    /// Bumped once per job; workers wait for it to change.
    epoch: AtomicU32,
    /// `(epoch << 32) | next task index`, so a worker that is late for job `e` cannot claim a task of
    /// job `e + 1` (the epoch part would not match).
    claim: AtomicU64,
    done: AtomicUsize,
    job: Mutex<Job>,
    /// Serializes submitters; a second caller that finds it taken runs its tasks inline.
    submit: Mutex<()>,
    parked: AtomicUsize,
    sleepers: Mutex<()>,
    wake: Condvar,
}

fn noop(_task: usize) {}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok().and_then(|v| v.parse::<usize>().ok())
}

fn shared() -> &'static Shared {
    static S: OnceLock<&'static Shared> = OnceLock::new();
    S.get_or_init(|| {
        let threads = env_usize("PYTORCHES_CPU_THREADS")
            .filter(|&n| n > 0)
            .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
        let spin = Duration::from_micros(env_usize("PYTORCHES_CPU_SPIN_US").unwrap_or(100) as u64);
        let s: &'static Shared = Box::leak(Box::new(Shared {
            threads,
            spin,
            epoch: AtomicU32::new(0),
            claim: AtomicU64::new(0),
            done: AtomicUsize::new(0),
            job: Mutex::new(Job { f: &noop as &(dyn Fn(usize) + Sync) as *const (dyn Fn(usize) + Sync), n: 0, epoch: 0 }),
            submit: Mutex::new(()),
            parked: AtomicUsize::new(0),
            sleepers: Mutex::new(()),
            wake: Condvar::new(),
        }));
        for i in 1..threads {
            let _ = std::thread::Builder::new().name(format!("pytorches-cpu-{i}")).spawn(move || worker(s));
        }
        s
    })
}

pub fn threads() -> usize {
    shared().threads
}

/// Claims and runs tasks of job `epoch` until none are left. Returns how many this thread ran.
fn drain(s: &Shared, job: Job) -> usize {
    let mut ran = 0;
    loop {
        let cur = s.claim.load(Ordering::Acquire);
        if (cur >> 32) as u32 != job.epoch || (cur & 0xFFFF_FFFF) as usize >= job.n {
            return ran;
        }
        if s.claim.compare_exchange_weak(cur, cur + 1, Ordering::AcqRel, Ordering::Relaxed).is_ok() {
            unsafe { (*job.f)((cur & 0xFFFF_FFFF) as usize) };
            s.done.fetch_add(1, Ordering::Release);
            ran += 1;
        }
    }
}

fn worker(s: &'static Shared) {
    let mut seen = s.epoch.load(Ordering::Acquire);
    loop {
        // Spin for the configured window, then sleep until a new job arrives.
        let start = Instant::now();
        let mut spins = 0u32;
        while s.epoch.load(Ordering::Acquire) == seen {
            std::hint::spin_loop();
            spins += 1;
            if spins % 128 == 0 && start.elapsed() >= s.spin {
                let mut g = s.sleepers.lock().unwrap_or_else(|p| p.into_inner());
                s.parked.fetch_add(1, Ordering::SeqCst);
                while s.epoch.load(Ordering::SeqCst) == seen {
                    g = s.wake.wait(g).unwrap_or_else(|p| p.into_inner());
                }
                s.parked.fetch_sub(1, Ordering::SeqCst);
                break;
            }
        }
        let job = *s.job.lock().unwrap_or_else(|p| p.into_inner());
        seen = job.epoch;
        drain(s, job);
    }
}

/// Runs `f(task)` for every task in `0..tasks`, in parallel when there is more than one. Returns
/// when all have finished. The calling thread works too.
pub fn run_tasks(tasks: usize, f: impl Fn(usize) + Sync) {
    if tasks <= 1 || threads() == 1 {
        for t in 0..tasks {
            f(t);
        }
        return;
    }
    let s = shared();
    // Another thread is already using the pool (e.g. concurrent ops from Python threads): run inline
    // rather than queue behind it.
    let Ok(_submit) = s.submit.try_lock() else {
        for t in 0..tasks {
            f(t);
        }
        return;
    };
    let fref: &(dyn Fn(usize) + Sync) = &f;
    // SAFETY: we wait below until all `tasks` tasks are done, so `f` outlives every call through this.
    let fptr: *const (dyn Fn(usize) + Sync) = unsafe { std::mem::transmute(fref) };
    let epoch = s.epoch.load(Ordering::Relaxed).wrapping_add(1);
    let job = Job { f: fptr, n: tasks, epoch };
    *s.job.lock().unwrap_or_else(|p| p.into_inner()) = job;
    s.done.store(0, Ordering::Relaxed);
    s.claim.store((epoch as u64) << 32, Ordering::Release);
    s.epoch.store(epoch, Ordering::SeqCst);
    if s.parked.load(Ordering::SeqCst) > 0 {
        // Take the lock so a worker between "saw the old epoch" and "started waiting" cannot miss this.
        drop(s.sleepers.lock().unwrap_or_else(|p| p.into_inner()));
        s.wake.notify_all();
    }
    drain(s, job);
    let mut spins = 0u32;
    while s.done.load(Ordering::Acquire) < tasks {
        std::hint::spin_loop();
        spins += 1;
        if spins % 4096 == 0 {
            std::thread::yield_now();
        }
    }
}

/// Splits `0..n` into chunks of `chunk` and runs `f(lo, hi)` on each.
pub fn for_chunks(n: usize, chunk: usize, f: impl Fn(usize, usize) + Sync) {
    let tasks = n.div_ceil(chunk.max(1));
    run_tasks(tasks, |t| f(t * chunk, ((t + 1) * chunk).min(n)));
}

/// A raw pointer that may be shared across tasks that write disjoint regions.
#[derive(Clone, Copy)]
pub struct SendPtr<T>(pub *mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

#[derive(Clone, Copy)]
pub struct SendConst<T>(pub *const T);
unsafe impl<T> Send for SendConst<T> {}
unsafe impl<T> Sync for SendConst<T> {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn every_task_runs_exactly_once() {
        for tasks in [0usize, 1, 2, 7, 64, 1000] {
            let hits: Vec<AtomicUsize> = (0..tasks).map(|_| AtomicUsize::new(0)).collect();
            run_tasks(tasks, |t| {
                hits[t].fetch_add(1, Ordering::SeqCst);
            });
            assert!(hits.iter().all(|h| h.load(Ordering::SeqCst) == 1), "tasks={tasks}");
        }
    }

    #[test]
    fn many_jobs_back_to_back_and_after_pauses() {
        let total = AtomicUsize::new(0);
        for round in 0..3000 {
            run_tasks(16, |_| {
                total.fetch_add(1, Ordering::SeqCst);
            });
            if round % 500 == 0 {
                std::thread::sleep(Duration::from_millis(3)); // long enough for workers to park
            }
        }
        assert_eq!(total.load(Ordering::SeqCst), 3000 * 16);
    }

    #[test]
    fn concurrent_callers_do_not_deadlock_or_lose_tasks() {
        let total = AtomicUsize::new(0);
        std::thread::scope(|sc| {
            for _ in 0..4 {
                sc.spawn(|| {
                    for _ in 0..500 {
                        run_tasks(8, |_| {
                            total.fetch_add(1, Ordering::SeqCst);
                        });
                    }
                });
            }
        });
        assert_eq!(total.load(Ordering::SeqCst), 4 * 500 * 8);
    }

    /// Dispatch latency of the pool (run with `--ignored --nocapture`): empty tasks, back to back and
    /// after an idle gap, which separates "workers still spinning" from "workers asleep".
    #[test]
    #[ignore = "timing probe"]
    fn dispatch_latency() {
        for tasks in [1usize, 4, 16, 64] {
            run_tasks(tasks, |_| {});
            let n = 20_000;
            let t = Instant::now();
            for _ in 0..n {
                run_tasks(tasks, |_| {});
            }
            let hot = t.elapsed().as_secs_f64() / n as f64 * 1e6;
            let n = 200;
            let mut lat = Vec::with_capacity(n);
            for _ in 0..n {
                std::thread::sleep(Duration::from_micros(300));
                let t = Instant::now();
                run_tasks(tasks, |_| {});
                lat.push(t.elapsed().as_secs_f64() * 1e6);
            }
            lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let slow = lat.iter().filter(|&&l| l > 500.0).count();
            eprintln!(
                "tasks {tasks:>3}: back-to-back {hot:6.1} us/call; after a pause: median {:6.1}, p99 {:7.1}, max {:9.1} us, {slow} calls over 500 us",
                lat[n / 2],
                lat[n * 99 / 100],
                lat[n - 1]
            );
        }
    }
}
