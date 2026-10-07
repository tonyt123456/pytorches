//! Sum over one axis of an `(outer, n, inner)` contiguous tensor.
//!
//! Results do not depend on the thread count: each reduction is cut into fixed-size blocks, blocks are
//! summed independently, and the partial sums are combined in a fixed order.

use crate::pool::{SendConst, SendPtr, run_tasks};
use crate::simd;

/// Elements summed by one task when reducing a long contiguous row.
const BLOCK: usize = 1 << 16;
/// Columns handled by one task when reducing across rows (`inner > 1`).
const COLS: usize = 2048;

#[inline(always)]
fn sum_slice_plain(x: &[f32]) -> f32 {
    // Several independent accumulators let the compiler vectorize and hide add latency.
    let mut acc = [0f32; 16];
    let mut chunks = x.chunks_exact(16);
    for c in &mut chunks {
        for i in 0..16 {
            acc[i] += c[i];
        }
    }
    let mut s = 0.0;
    for v in acc {
        s += v;
    }
    for &v in chunks.remainder() {
        s += v;
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sum_slice_avx2(x: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    unsafe {
        let n = x.len();
        let p = x.as_ptr();
        let (mut a0, mut a1, mut a2, mut a3) = (_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps());
        let n32 = n / 32 * 32;
        let mut i = 0;
        while i < n32 {
            a0 = _mm256_add_ps(a0, _mm256_loadu_ps(p.add(i)));
            a1 = _mm256_add_ps(a1, _mm256_loadu_ps(p.add(i + 8)));
            a2 = _mm256_add_ps(a2, _mm256_loadu_ps(p.add(i + 16)));
            a3 = _mm256_add_ps(a3, _mm256_loadu_ps(p.add(i + 24)));
            i += 32;
        }
        let v = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
        let lo = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
        let sh = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
        let sh = _mm_add_ss(sh, _mm_shuffle_ps::<1>(sh, sh));
        let mut s = _mm_cvtss_f32(sh);
        for &v in &x[n32..] {
            s += v;
        }
        s
    }
}

fn sum_slice(x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if simd::has_avx2() {
        return unsafe { sum_slice_avx2(x) };
    }
    sum_slice_plain(x)
}

#[inline(always)]
unsafe fn add_rows_impl(dst: *mut f32, src: *const f32, len: usize) {
    unsafe {
        let d = std::slice::from_raw_parts_mut(dst, len);
        let s = std::slice::from_raw_parts(src, len);
        for i in 0..len {
            d[i] += s[i];
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn add_rows_avx2(dst: *mut f32, src: *const f32, len: usize) {
    unsafe { add_rows_impl(dst, src, len) }
}

unsafe fn add_row(dst: *mut f32, src: *const f32, len: usize, avx: bool) {
    unsafe {
        #[cfg(target_arch = "x86_64")]
        if avx {
            return add_rows_avx2(dst, src, len);
        }
        let _ = avx;
        add_rows_impl(dst, src, len)
    }
}

/// `out[o * inner + i] = sum_j x[(o * n + j) * inner + i]`.
pub unsafe fn sum_axis(x: *const f32, outer: usize, n: usize, inner: usize, out: *mut f32) {
    let total = outer * inner;
    if total == 0 {
        return;
    }
    let (xs, os) = (SendConst(x), SendPtr(out));
    if n == 0 {
        unsafe { std::slice::from_raw_parts_mut(out, total).fill(0.0) };
        return;
    }
    if inner == 1 {
        // Reduce contiguous rows. Long rows are split into blocks so one row can use every core.
        let blocks = n.div_ceil(BLOCK);
        let mut partial = vec![0f32; outer * blocks];
        let ps = SendPtr(partial.as_mut_ptr());
        run_tasks(outer * blocks, |t| unsafe {
            let (xs, ps) = (xs, ps);
            let (row, b) = (t / blocks, t % blocks);
            let lo = b * BLOCK;
            let hi = (lo + BLOCK).min(n);
            let s = sum_slice(std::slice::from_raw_parts(xs.0.add(row * n + lo), hi - lo));
            *ps.0.add(t) = s;
        });
        for row in 0..outer {
            let mut s = 0.0f32;
            for b in 0..blocks {
                s += partial[row * blocks + b];
            }
            unsafe { *os.0.add(row) = s };
        }
        return;
    }
    // inner > 1: out[o][i] accumulates the rows j in order, one column chunk per task.
    let avx = simd::has_avx2();
    let chunks = inner.div_ceil(COLS);
    run_tasks(outer * chunks, |t| unsafe {
        let (xs, os) = (xs, os);
        let (o, c) = (t / chunks, t % chunks);
        let i0 = c * COLS;
        let len = (i0 + COLS).min(inner) - i0;
        let dst = os.0.add(o * inner + i0);
        std::slice::from_raw_parts_mut(dst, len).fill(0.0);
        for j in 0..n {
            add_row(dst, xs.0.add((o * n + j) * inner + i0), len, avx);
        }
    });
}
