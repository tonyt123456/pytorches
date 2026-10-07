//! Single-precision matrix multiply, `C[m,n] = A[m,k] * B[k,n]` with arbitrary operand strides.
//!
//! BLIS-style: the problem is cut into cache blocks, A and B blocks are packed into contiguous
//! panels (so any strides, including transposes, cost nothing extra), and a register-tiled
//! microkernel (6x16 with AVX2/FMA, a scalar fallback elsewhere) does the arithmetic. The m x n
//! output is split into tiles that run as parallel tasks; the k loop always runs in order inside a
//! task, so results do not depend on the thread count.

use crate::pool::{SendPtr, run_tasks, threads};
use crate::simd;
use std::cell::RefCell;

const MR: usize = 6;
const NR: usize = 16;
/// Depth of a k block: a packed B panel (KC x NR) stays in L1.
const KC: usize = 256;
/// Rows of an A block: the packed block (MC x KC) stays in L2.
const MC: usize = 96;
/// Columns of a B block: the packed block (KC x NC) lives in the shared cache.
const NC: usize = 2048;
/// Below this many multiply-adds the packing and threading overhead is not worth it.
const SMALL: usize = 1 << 15;
/// Packed B blocks smaller than this many floats are packed without the pool.
const PACK_PAR_MIN: usize = 1 << 16;
/// Multiply-adds one task should have at least, so tiny problems do not fan out to every core.
const TASK_MACS: usize = 1 << 18;

/// A matrix operand: element `(i, j)` is `ptr[i * rs + j * cs]`.
#[derive(Clone, Copy)]
pub struct Mat {
    pub ptr: *const f32,
    pub rs: usize,
    pub cs: usize,
}

// Tasks only read through the operand pointers; the output tiles they write are disjoint.
unsafe impl Send for Mat {}
unsafe impl Sync for Mat {}

impl Mat {
    #[inline(always)]
    unsafe fn at(&self, i: usize, j: usize) -> f32 {
        unsafe { *self.ptr.add(i * self.rs + j * self.cs) }
    }
}

/// `c` is row-major with leading dimension `n`.
pub unsafe fn sgemm(m: usize, n: usize, k: usize, a: Mat, b: Mat, c: *mut f32) {
    unsafe {
        if m == 0 || n == 0 {
            return;
        }
        if k == 0 {
            std::slice::from_raw_parts_mut(c, m * n).fill(0.0);
            return;
        }
        if m * n * k < SMALL {
            return naive(m, n, k, a, b, c);
        }
        let avx = simd::has_avx2();
        let nc_max = NC.min(n.div_ceil(NR) * NR);
        let kc_max = KC.min(k);
        let mut bpack = vec![0f32; kc_max * nc_max];
        let bp = SendPtr(bpack.as_mut_ptr());
        let c_s = SendPtr(c);

        for jc in (0..n).step_by(NC) {
            let nc = NC.min(n - jc);
            let npanels = nc.div_ceil(NR);
            for pc in (0..k).step_by(KC) {
                let kc = KC.min(k - pc);
                let accumulate = pc > 0;

                // Pack the B block (kc x nc) into NR-wide panels. Small blocks are packed on this thread:
                // handing them to the pool costs more than the copy.
                let pack = |jp: usize| {
                    let bp = bp;
                    let j0 = jc + jp * NR;
                    pack_b(b, pc, j0, kc, NR.min(jc + nc - j0), bp.0.add(jp * kc * NR));
                };
                if npanels * kc * NR < PACK_PAR_MIN {
                    (0..npanels).for_each(pack);
                } else {
                    let per = 4;
                    run_tasks(npanels.div_ceil(per), |t| {
                        for jp in t * per..((t + 1) * per).min(npanels) {
                            pack(jp);
                        }
                    });
                }

                // Tiles: MC-row blocks times slices of B panels. Aim for a few tasks per core so work
                // stealing can balance fast and slow cores, but no more than the work justifies, and
                // keep slices at least two panels wide so repeated A packing stays a small overhead.
                let mblocks = m.div_ceil(MC);
                let want = (threads() * 4).min((m * nc * kc / TASK_MACS).max(1));
                let nslices = want.div_ceil(mblocks).clamp(1, (npanels / 2).max(1));
                let slice_panels = npanels.div_ceil(nslices);
                let nslices = npanels.div_ceil(slice_panels);
                run_tasks(mblocks * nslices, |t| {
                    let (bp, c_s) = (bp, c_s);
                    let (ib, sl) = (t / nslices, t % nslices);
                    let ic = ib * MC;
                    let mc = MC.min(m - ic);
                    APACK.with(|cell| {
                        let mut buf = cell.borrow_mut();
                        if buf.len() < MC * KC {
                            buf.resize(MC * KC, 0.0);
                        }
                        let ap = buf.as_mut_ptr();
                        pack_a(a, ic, pc, mc, kc, ap);
                        let (jp0, jp1) = (sl * slice_panels, ((sl + 1) * slice_panels).min(npanels));
                        for jp in jp0..jp1 {
                            let j0 = jc + jp * NR;
                            let nr = NR.min(jc + nc - j0);
                            let bpanel = bp.0.add(jp * kc * NR);
                            for ir in (0..mc).step_by(MR) {
                                let mr = MR.min(mc - ir);
                                let cptr = c_s.0.add((ic + ir) * n + j0);
                                micro(avx, kc, ap.add(ir / MR * kc * MR), bpanel, cptr, n, mr, nr, accumulate);
                            }
                        }
                    });
                });
            }
        }
    }
}

thread_local! {
    static APACK: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Plain triple loop for tiny problems.
unsafe fn naive(m: usize, n: usize, k: usize, a: Mat, b: Mat, c: *mut f32) {
    unsafe {
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for p in 0..k {
                    s += a.at(i, p) * b.at(p, j);
                }
                *c.add(i * n + j) = s;
            }
        }
    }
}

/// Packs rows `ic..ic+mc`, columns `pc..pc+kc` of A into panels of MR rows: panel `p` holds, for each
/// `kk`, the MR values of that column (zero-padded below the last row).
unsafe fn pack_a(a: Mat, ic: usize, pc: usize, mc: usize, kc: usize, dst: *mut f32) {
    unsafe {
        for p in 0..mc.div_ceil(MR) {
            let rows = MR.min(mc - p * MR);
            let d = dst.add(p * kc * MR);
            if a.rs == 1 {
                // Transposed A: consecutive rows are adjacent in memory, walk k outermost.
                for kk in 0..kc {
                    for i in 0..MR {
                        *d.add(kk * MR + i) = if i < rows { a.at(ic + p * MR + i, pc + kk) } else { 0.0 };
                    }
                }
            } else {
                for i in 0..MR {
                    if i < rows {
                        let src = a.ptr.add((ic + p * MR + i) * a.rs + pc * a.cs);
                        for kk in 0..kc {
                            *d.add(kk * MR + i) = *src.add(kk * a.cs);
                        }
                    } else {
                        for kk in 0..kc {
                            *d.add(kk * MR + i) = 0.0;
                        }
                    }
                }
            }
        }
    }
}

/// Packs rows `pc..pc+kc` and columns `j0..j0+nr` of B into one NR-wide panel (zero-padded to NR).
unsafe fn pack_b(b: Mat, pc: usize, j0: usize, kc: usize, nr: usize, dst: *mut f32) {
    unsafe {
        if b.cs == 1 && nr == NR {
            for kk in 0..kc {
                std::ptr::copy_nonoverlapping(b.ptr.add((pc + kk) * b.rs + j0), dst.add(kk * NR), NR);
            }
        } else if b.rs == 1 {
            // Transposed B: for a fixed column the k values are adjacent; walk columns outermost.
            for j in 0..NR {
                if j < nr {
                    let src = b.ptr.add(pc + (j0 + j) * b.cs);
                    for kk in 0..kc {
                        *dst.add(kk * NR + j) = *src.add(kk);
                    }
                } else {
                    for kk in 0..kc {
                        *dst.add(kk * NR + j) = 0.0;
                    }
                }
            }
        } else {
            for kk in 0..kc {
                for j in 0..NR {
                    *dst.add(kk * NR + j) = if j < nr { b.at(pc + kk, j0 + j) } else { 0.0 };
                }
            }
        }
    }
}

/// Computes an `mr x nr` tile of C from one packed A panel and one packed B panel.
#[inline(always)]
unsafe fn micro(avx: bool, kc: usize, ap: *const f32, bp: *const f32, c: *mut f32, ldc: usize, mr: usize, nr: usize, acc: bool) {
    unsafe {
        #[cfg(target_arch = "x86_64")]
        if avx {
            return micro_avx2(kc, ap, bp, c, ldc, mr, nr, acc);
        }
        let _ = avx;
        micro_plain(kc, ap, bp, c, ldc, mr, nr, acc)
    }
}

unsafe fn micro_plain(kc: usize, ap: *const f32, bp: *const f32, c: *mut f32, ldc: usize, mr: usize, nr: usize, acc: bool) {
    unsafe {
        let mut t = [[0f32; NR]; MR];
        for p in 0..kc {
            for i in 0..MR {
                let av = *ap.add(p * MR + i);
                for j in 0..NR {
                    t[i][j] += av * *bp.add(p * NR + j);
                }
            }
        }
        for i in 0..mr {
            for j in 0..nr {
                let d = c.add(i * ldc + j);
                *d = if acc { *d + t[i][j] } else { t[i][j] };
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn micro_avx2(kc: usize, ap: *const f32, bp: *const f32, c: *mut f32, ldc: usize, mr: usize, nr: usize, acc: bool) {
    use std::arch::x86_64::*;
    unsafe {
        let mut t = [[_mm256_setzero_ps(); 2]; MR];
        let (mut a, mut b) = (ap, bp);
        for _ in 0..kc {
            let b0 = _mm256_loadu_ps(b);
            let b1 = _mm256_loadu_ps(b.add(8));
            for i in 0..MR {
                let av = _mm256_broadcast_ss(&*a.add(i));
                t[i][0] = _mm256_fmadd_ps(av, b0, t[i][0]);
                t[i][1] = _mm256_fmadd_ps(av, b1, t[i][1]);
            }
            a = a.add(MR);
            b = b.add(NR);
        }
        if mr == MR && nr == NR {
            for i in 0..MR {
                let row = c.add(i * ldc);
                if acc {
                    _mm256_storeu_ps(row, _mm256_add_ps(_mm256_loadu_ps(row), t[i][0]));
                    _mm256_storeu_ps(row.add(8), _mm256_add_ps(_mm256_loadu_ps(row.add(8)), t[i][1]));
                } else {
                    _mm256_storeu_ps(row, t[i][0]);
                    _mm256_storeu_ps(row.add(8), t[i][1]);
                }
            }
        } else {
            let mut buf = [0f32; MR * NR];
            for i in 0..MR {
                _mm256_storeu_ps(buf.as_mut_ptr().add(i * NR), t[i][0]);
                _mm256_storeu_ps(buf.as_mut_ptr().add(i * NR + 8), t[i][1]);
            }
            for i in 0..mr {
                for j in 0..nr {
                    let d = c.add(i * ldc + j);
                    *d = if acc { *d + buf[i * NR + j] } else { buf[i * NR + j] };
                }
            }
        }
    }
}
