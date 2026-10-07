//! Elementwise ops: unary, binary (with broadcasting and strides), axpy, fill, random, copy.
//!
//! Operands are walked in place: dimensions are coalesced, the output range is cut into fixed chunks
//! that run in parallel, and each chunk is processed as rows whose inner stride is 1 (contiguous),
//! 0 (broadcast) or anything else (gathered). Inner loops are compiled twice, once with AVX2/FMA and
//! once for the baseline target, and picked at run time.

use crate::pool::{SendConst, SendPtr, for_chunks, run_tasks};
use crate::simd;
use pytorches_plugin_abi::op::*;

/// Elements per parallel task. Large enough to amortize scheduling, small enough to balance cores.
const CHUNK: usize = 1 << 16;
/// Below this many elements an op runs on the calling thread.
const PAR_MIN: usize = 1 << 15;

/// One dimension after coalescing: its size and the element stride of each of the `N` operands.
type Dim<const N: usize> = (usize, [usize; N]);

/// Drops size-1 dimensions and merges neighbours that are contiguous for every operand
/// (`strides[k]` is operand k's strides). The output is contiguous.
pub fn coalesce<const N: usize>(shape: &[u64], strides: [&[u64]; N]) -> Vec<Dim<N>> {
    let mut v: Vec<Dim<N>> = Vec::new();
    for (d, &s) in shape.iter().enumerate() {
        let s = s as usize;
        if s == 1 {
            continue;
        }
        let mut st = [0usize; N];
        for k in 0..N {
            st[k] = strides[k][d] as usize;
        }
        if let Some(last) = v.last_mut() {
            if (0..N).all(|k| last.1[k] == st[k] * s) {
                last.0 *= s;
                last.1 = st;
                continue;
            }
        }
        v.push((s, st));
    }
    v
}

/// Calls `row(out_offset, operand_offsets, len)` for each maximal inner run of output range `[lo, hi)`.
fn rows<const N: usize>(dims: &[Dim<N>], lo: usize, hi: usize, mut row: impl FnMut(usize, [usize; N], usize)) {
    if dims.is_empty() {
        row(lo, [0; N], hi - lo);
        return;
    }
    let nd = dims.len();
    let mut idx = vec![0usize; nd];
    let mut rem = lo;
    for d in (0..nd).rev() {
        idx[d] = rem % dims[d].0;
        rem /= dims[d].0;
    }
    let mut offs = [0usize; N];
    for d in 0..nd {
        for k in 0..N {
            offs[k] += idx[d] * dims[d].1[k];
        }
    }
    let inner = dims[nd - 1].0;
    let mut pos = lo;
    while pos < hi {
        let len = (inner - idx[nd - 1]).min(hi - pos);
        row(pos, offs, len);
        pos += len;
        idx[nd - 1] += len;
        for k in 0..N {
            offs[k] += len * dims[nd - 1].1[k];
        }
        let mut d = nd - 1;
        while d > 0 && idx[d] == dims[d].0 {
            for k in 0..N {
                offs[k] -= dims[d].0 * dims[d].1[k];
            }
            idx[d] = 0;
            d -= 1;
            idx[d] += 1;
            for k in 0..N {
                offs[k] += dims[d].1[k];
            }
        }
    }
}

fn inner_strides<const N: usize>(dims: &[Dim<N>]) -> [usize; N] {
    dims.last().map(|d| d.1).unwrap_or([0; N])
}

// ---- unary ------------------------------------------------------------------------------------

#[inline(always)]
fn unary_generic(op: u32, x: &[f32], o: &mut [f32]) {
    match op {
        NEG => o.iter_mut().zip(x).for_each(|(o, &v)| *o = -v),
        RELU => o.iter_mut().zip(x).for_each(|(o, &v)| *o = v.max(0.0)),
        STEP => o.iter_mut().zip(x).for_each(|(o, &v)| *o = if v > 0.0 { 1.0 } else { 0.0 }),
        EXP => o.iter_mut().zip(x).for_each(|(o, &v)| *o = v.exp()),
        LOG => o.iter_mut().zip(x).for_each(|(o, &v)| *o = v.ln()),
        TANH => o.iter_mut().zip(x).for_each(|(o, &v)| *o = v.tanh()),
        _ => o.copy_from_slice(x), // COPY
    }
}

fn unary_plain(op: u32, x: &[f32], o: &mut [f32]) {
    unary_generic(op, x, o)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn unary_avx2(op: u32, x: &[f32], o: &mut [f32]) {
    unsafe {
        use std::arch::x86_64::*;
        use simd::x86::*;
        let n = x.len().min(o.len());
        let n8 = n / 8 * 8;
        let (xp, op_) = (x.as_ptr(), o.as_mut_ptr());
        macro_rules! vec_loop {
            ($f:expr) => {{
                let mut i = 0;
                while i < n8 {
                    _mm256_storeu_ps(op_.add(i), $f(_mm256_loadu_ps(xp.add(i))));
                    i += 8;
                }
            }};
        }
        match op {
            EXP => vec_loop!(|v| exp256(v)),
            LOG => vec_loop!(|v| log256(v)),
            TANH => vec_loop!(|v| tanh256(v)),
            _ => return unary_generic(op, x, o),
        }
        unary_generic(op, &x[n8..n], &mut o[n8..n]);
    }
}

fn unary_dispatch(op: u32, x: &[f32], o: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if simd::has_avx2() {
        return unsafe { unary_avx2(op, x, o) };
    }
    unary_plain(op, x, o)
}

/// Applies a unary op (or a strided COPY) from `x` (shape/strides given) into contiguous `out`.
pub unsafe fn unary(op: u32, x: *const f32, shape: &[u64], strides: &[u64], out: *mut f32) {
    let n: usize = shape.iter().map(|&s| s as usize).product();
    if n == 0 {
        return;
    }
    let dims = coalesce(shape, [strides]);
    if op == COPY && is_plain_transpose(&dims) {
        // Source is a contiguous [rows, cols] matrix; the output is its [cols, rows] transpose.
        return unsafe { transpose(x, dims[1].0, dims[0].0, out) };
    }
    let (xs, os) = (SendConst(x), SendPtr(out));
    let chunk = if n < PAR_MIN { n } else { CHUNK };
    for_chunks(n, chunk, |lo, hi| unsafe {
        let (xs, os) = (xs, os);
        rows(&dims, lo, hi, |oo, offs, len| {
            let dst = std::slice::from_raw_parts_mut(os.0.add(oo), len);
            let st = inner_strides(&dims)[0];
            if st == 1 {
                unary_dispatch(op, std::slice::from_raw_parts(xs.0.add(offs[0]), len), dst);
            } else if st == 0 {
                let v = [*xs.0.add(offs[0])];
                let mut tmp = [0f32];
                unary_dispatch(op, &v, &mut tmp);
                dst.fill(tmp[0]);
            } else {
                // Gather a block at a time so the math still runs on contiguous data.
                let mut buf = [0f32; 256];
                let mut done = 0;
                while done < len {
                    let m = (len - done).min(256);
                    for i in 0..m {
                        buf[i] = *xs.0.add(offs[0] + (done + i) * st);
                    }
                    unary_dispatch(op, &buf[..m], &mut dst[done..done + m]);
                    done += m;
                }
            }
        });
    });
}

/// A COPY that transposes a contiguous `[r, c]` matrix into `[c, r]` has the merged dims
/// `(c, [1])`, `(r, [c])`: the outer output dim walks the source with stride 1, the inner with stride c.
fn is_plain_transpose(dims: &[Dim<1>]) -> bool {
    dims.len() == 2 && dims[0].1[0] == 1 && dims[1].1[0] == dims[0].0
}

/// Cache-blocked parallel transpose: `src` is a contiguous `[rows, cols]` matrix, `out` becomes `[cols, rows]`.
pub unsafe fn transpose(src: *const f32, rows: usize, cols: usize, out: *mut f32) {
    const T: usize = 32;
    let (s, o) = (SendConst(src), SendPtr(out));
    let tile_rows = rows.div_ceil(T);
    let work = rows * cols;
    let tasks = if work < PAR_MIN { 1 } else { tile_rows };
    let per = tile_rows.div_ceil(tasks);
    run_tasks(tasks, |t| unsafe {
        let (s, o) = (s, o);
        for tr in t * per..((t + 1) * per).min(tile_rows) {
            let i0 = tr * T;
            let i1 = (i0 + T).min(rows);
            for j0 in (0..cols).step_by(T) {
                let j1 = (j0 + T).min(cols);
                for i in i0..i1 {
                    for j in j0..j1 {
                        *o.0.add(j * rows + i) = *s.0.add(i * cols + j);
                    }
                }
            }
        }
    });
}

// ---- binary -----------------------------------------------------------------------------------

/// One row of a binary op; `sa`/`sb` are the operands' inner strides (1 = contiguous, 0 = broadcast).
#[inline(always)]
unsafe fn binary_row_impl(op: u32, a: *const f32, sa: usize, b: *const f32, sb: usize, o: *mut f32, len: usize) {
    unsafe {
        macro_rules! go {
            ($f:expr) => {{
                match (sa, sb) {
                    (1, 1) => {
                        let (a, b, o) = (std::slice::from_raw_parts(a, len), std::slice::from_raw_parts(b, len), std::slice::from_raw_parts_mut(o, len));
                        for i in 0..len {
                            o[i] = $f(a[i], b[i]);
                        }
                    }
                    (1, 0) => {
                        let y = *b;
                        let (a, o) = (std::slice::from_raw_parts(a, len), std::slice::from_raw_parts_mut(o, len));
                        for i in 0..len {
                            o[i] = $f(a[i], y);
                        }
                    }
                    (0, 1) => {
                        let x = *a;
                        let (b, o) = (std::slice::from_raw_parts(b, len), std::slice::from_raw_parts_mut(o, len));
                        for i in 0..len {
                            o[i] = $f(x, b[i]);
                        }
                    }
                    _ => {
                        for i in 0..len {
                            *o.add(i) = $f(*a.add(i * sa), *b.add(i * sb));
                        }
                    }
                }
            }};
        }
        match op {
            ADD => go!(|x: f32, y: f32| x + y),
            SUB => go!(|x: f32, y: f32| x - y),
            MUL => go!(|x: f32, y: f32| x * y),
            _ => go!(|x: f32, y: f32| x / y),
        }
    }
}

unsafe fn binary_row_plain(op: u32, a: *const f32, sa: usize, b: *const f32, sb: usize, o: *mut f32, len: usize) {
    unsafe { binary_row_impl(op, a, sa, b, sb, o, len) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn binary_row_avx2(op: u32, a: *const f32, sa: usize, b: *const f32, sb: usize, o: *mut f32, len: usize) {
    unsafe { binary_row_impl(op, a, sa, b, sb, o, len) }
}

/// out = a (op) b over `shape`, each input with its own (possibly broadcast, stride-0) strides.
pub unsafe fn binary(op: u32, a: *const f32, sa: &[u64], b: *const f32, sb: &[u64], shape: &[u64], out: *mut f32) {
    let n: usize = shape.iter().map(|&s| s as usize).product();
    if n == 0 {
        return;
    }
    let dims = coalesce(shape, [sa, sb]);
    let [ia, ib] = inner_strides(&dims);
    let avx = simd::has_avx2();
    let (ap, bp, os) = (SendConst(a), SendConst(b), SendPtr(out));
    let chunk = if n < PAR_MIN { n } else { CHUNK };
    for_chunks(n, chunk, |lo, hi| unsafe {
        let (ap, bp, os) = (ap, bp, os);
        rows(&dims, lo, hi, |oo, offs, len| {
            let (pa, pb, po) = (ap.0.add(offs[0]), bp.0.add(offs[1]), os.0.add(oo));
            #[cfg(target_arch = "x86_64")]
            if avx {
                return binary_row_avx2(op, pa, ia, pb, ib, po, len);
            }
            let _ = avx;
            binary_row_plain(op, pa, ia, pb, ib, po, len)
        });
    });
}

// ---- axpy, fill, random -------------------------------------------------------------------------

#[inline(always)]
unsafe fn axpy_impl(a: *const f32, b: *const f32, o: *mut f32, alpha: f32, len: usize) {
    unsafe {
        // `o` may be the same buffer as `a` (in-place update), so go through raw pointers elementwise.
        for i in 0..len {
            *o.add(i) = *a.add(i) + alpha * *b.add(i);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn axpy_avx2(a: *const f32, b: *const f32, o: *mut f32, alpha: f32, len: usize) {
    use std::arch::x86_64::*;
    unsafe {
        let va = _mm256_set1_ps(alpha);
        let n8 = len / 8 * 8;
        let mut i = 0;
        while i < n8 {
            let r = _mm256_fmadd_ps(va, _mm256_loadu_ps(b.add(i)), _mm256_loadu_ps(a.add(i)));
            _mm256_storeu_ps(o.add(i), r);
            i += 8;
        }
        axpy_impl(a.add(n8), b.add(n8), o.add(n8), alpha, len - n8);
    }
}

/// out = a + alpha * b over `n` contiguous elements; `out` may equal `a`.
pub unsafe fn axpy(a: *const f32, b: *const f32, out: *mut f32, alpha: f32, n: usize) {
    let (ap, bp, os) = (SendConst(a), SendConst(b), SendPtr(out));
    let avx = simd::has_avx2();
    let chunk = if n < PAR_MIN { n.max(1) } else { CHUNK };
    for_chunks(n, chunk, |lo, hi| unsafe {
        let (ap, bp, os) = (ap, bp, os);
        #[cfg(target_arch = "x86_64")]
        if avx {
            return axpy_avx2(ap.0.add(lo), bp.0.add(lo), os.0.add(lo), alpha, hi - lo);
        }
        let _ = avx;
        axpy_impl(ap.0.add(lo), bp.0.add(lo), os.0.add(lo), alpha, hi - lo)
    });
}

pub unsafe fn fill(out: *mut f32, n: usize, v: f32) {
    let os = SendPtr(out);
    let chunk = if n < PAR_MIN { n.max(1) } else { CHUNK };
    for_chunks(n, chunk, |lo, hi| unsafe {
        let os = os;
        std::slice::from_raw_parts_mut(os.0.add(lo), hi - lo).fill(v);
    });
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Counter-based N(0,1); the exact recipe is specified in `pytorches_plugin_abi::op::RAND_NORMAL`.
fn rand_normal(seed: u64, i: u64) -> f32 {
    let h = splitmix64(seed ^ i.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let u1 = ((h >> 40) + 1) as f32 / 16777216.0;
    let u2 = (h & 0xFF_FFFF) as f32 / 16777216.0;
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

pub unsafe fn randn(out: *mut f32, n: usize, seed: u64) {
    let os = SendPtr(out);
    let chunk = if n < PAR_MIN { n.max(1) } else { 1 << 14 };
    for_chunks(n, chunk, |lo, hi| unsafe {
        let os = os;
        for i in lo..hi {
            *os.0.add(i) = rand_normal(seed, i as u64);
        }
    });
}
