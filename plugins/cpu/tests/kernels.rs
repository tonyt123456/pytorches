//! CPU plugin kernels against plain reference loops, driven through the plugin ABI.

use pytorches_plugin_abi::*;
use pytorches_plugin_cpu::pytorches_plugin_entry;
use std::ffi::c_void;
use std::ptr::null_mut;

fn vt() -> &'static PluginVTable {
    unsafe { &*pytorches_plugin_entry() }
}

/// Deterministic values in [-1, 1).
fn data(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / 8388608.0) - 1.0
        })
        .collect()
}

struct Buf {
    ptr: *mut c_void,
    n: usize,
}

impl Buf {
    fn new(v: &[f32]) -> Buf {
        let mut ptr = null_mut();
        assert_eq!(unsafe { (vt().alloc)(0, v.len().max(1) * 4, &mut ptr) }, STATUS_OK);
        unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), ptr as *mut f32, v.len()) };
        Buf { ptr, n: v.len() }
    }
    fn empty(n: usize) -> Buf {
        Buf::new(&vec![f32::NAN; n])
    }
    fn get(&self) -> Vec<f32> {
        unsafe { std::slice::from_raw_parts(self.ptr as *const f32, self.n).to_vec() }
    }
}

impl Drop for Buf {
    fn drop(&mut self) {
        unsafe { (vt().free)(0, self.ptr) };
    }
}

fn contiguous(shape: &[u64]) -> Vec<u64> {
    let mut st = vec![0u64; shape.len()];
    let mut acc = 1;
    for i in (0..shape.len()).rev() {
        st[i] = acc;
        acc *= shape[i];
    }
    st
}

fn exec(code: u32, ints: [i64; 4], ins: &[(&Buf, &[u64], &[u64])], out: &Buf, oshape: &[u64]) -> Status {
    let ostr = contiguous(oshape);
    let desc = |b: &Buf, sh: &[u64], st: &[u64]| TensorDesc { data: b.ptr, dtype: DTYPE_F32, ndim: sh.len() as u32, shape: sh.as_ptr(), strides: st.as_ptr() };
    let ind: Vec<TensorDesc> = ins.iter().map(|(b, sh, st)| desc(b, sh, st)).collect();
    let od = desc(out, oshape, &ostr);
    let attrs = OpAttrs { ints };
    unsafe { (vt().execute)(0, code, &attrs, ind.as_ptr(), ind.len() as u32, &od, 1) }
}

fn close(what: &str, got: &[f32], want: &[f32], rtol: f32) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30);
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= rtol * scale || (g.is_nan() && w.is_nan()) || g == w, "{what}[{i}]: got {g}, want {w}");
    }
}

#[test]
fn matmul_all_transpose_combinations_and_odd_shapes() {
    for (m, k, n) in [(1, 1, 1), (5, 7, 3), (6, 16, 16), (7, 17, 33), (97, 255, 130), (100, 513, 200), (1, 300, 200), (300, 300, 1), (130, 1, 70), (250, 600, 250)] {
        for flags in 0..4i64 {
            let (ta, tb) = (flags & 1 != 0, flags & 2 != 0);
            // Stored shapes: A is [m,k] or [k,m]; B is [k,n] or [n,k].
            let sa: [u64; 2] = if ta { [k as u64, m as u64] } else { [m as u64, k as u64] };
            let sb: [u64; 2] = if tb { [n as u64, k as u64] } else { [k as u64, n as u64] };
            let (av, bv) = (data(m * k, 1), data(k * n, 2));
            let mut want = vec![0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut s = 0f64;
                    for p in 0..k {
                        let a = if ta { av[p * m + i] } else { av[i * k + p] };
                        let b = if tb { bv[j * k + p] } else { bv[p * n + j] };
                        s += a as f64 * b as f64;
                    }
                    want[i * n + j] = s as f32;
                }
            }
            let (a, b, out) = (Buf::new(&av), Buf::new(&bv), Buf::empty(m * n));
            let (sta, stb) = (contiguous(&sa), contiguous(&sb));
            let code = if flags == 0 { op::MATMUL } else { op::MATMUL_T };
            let st = exec(code, [flags, 0, 0, 0], &[(&a, &sa, &sta), (&b, &sb, &stb)], &out, &[m as u64, n as u64]);
            assert_eq!(st, STATUS_OK);
            close(&format!("matmul {m}x{k}x{n} flags {flags}"), &out.get(), &want, 2e-6);
        }
    }
}

#[test]
fn matmul_accepts_strided_inputs() {
    // A is a transposed view of a [k, m] matrix passed through plain MATMUL strides.
    let (m, k, n) = (13, 21, 17);
    let (av, bv) = (data(k * m, 3), data(k * n, 4));
    let mut want = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            want[i * n + j] = (0..k).map(|p| av[p * m + i] * bv[p * n + j]).sum();
        }
    }
    let (a, b, out) = (Buf::new(&av), Buf::new(&bv), Buf::empty(m * n));
    let (sa, sta) = ([m as u64, k as u64], [1u64, m as u64]);
    let (sb, stb) = ([k as u64, n as u64], contiguous(&[k as u64, n as u64]));
    assert_eq!(exec(op::MATMUL, [0; 4], &[(&a, &sa, &sta), (&b, &sb, &stb)], &out, &[m as u64, n as u64]), STATUS_OK);
    close("strided matmul", &out.get(), &want, 2e-6);
}

#[test]
fn binary_ops_broadcast_patterns() {
    let cases: [(&[u64], &[u64], &[u64]); 5] = [
        (&[37, 91], &[91, 1], &[37, 91]),      // row broadcast: [m,n] + [n]
        (&[37, 91], &[1, 0], &[37, 91]),       // column broadcast: [m,1] over n
        (&[37, 91], &[0, 0], &[37, 91]),       // scalar
        (&[4, 5, 6], &[30, 6, 1], &[4, 5, 6]), // full
        (&[300_000], &[1], &[300_000]),        // large, parallel
    ];
    for (shape, sa, _) in cases {
        for code in [op::ADD, op::SUB, op::MUL, op::DIV] {
            let n: usize = shape.iter().map(|&s| s as usize).product();
            let av = data(n.max(1) * 2, 5);
            let bv: Vec<f32> = data(n.max(1) * 2, 6).into_iter().map(|v| v + 1.5).collect();
            // Operand A takes the case's strides; operand B is always contiguous.
            let stb = contiguous(shape);
            let (a, b, out) = (Buf::new(&av), Buf::new(&bv), Buf::empty(n));
            assert_eq!(exec(code, [0; 4], &[(&a, shape, sa), (&b, shape, &stb)], &out, shape), STATUS_OK);
            // Reference: walk the multi-index.
            let nd = shape.len();
            let mut want = vec![0f32; n];
            for lin in 0..n {
                let (mut rem, mut off) = (lin, 0usize);
                for d in (0..nd).rev() {
                    off += (rem % shape[d] as usize) * sa[d] as usize;
                    rem /= shape[d] as usize;
                }
                let (x, y) = (av[off], bv[lin]);
                want[lin] = match code {
                    op::ADD => x + y,
                    op::SUB => x - y,
                    op::MUL => x * y,
                    _ => x / y,
                };
            }
            close(&format!("binary {code} {shape:?} strides {sa:?}"), &out.get(), &want, 1e-6);
        }
    }
}

#[test]
fn sum_axis_layouts() {
    for shape in [vec![1000u64], vec![33, 17], vec![5, 4, 3], vec![70_001], vec![3, 200_000], vec![100_000, 3], vec![2, 600, 5], vec![4, 4096, 70], vec![1, 1], vec![64, 4096]] {
        for axis in 0..shape.len() {
            let n: usize = shape.iter().map(|&s| s as usize).product();
            let xv = data(n, 7);
            let outer: usize = shape[..axis].iter().map(|&s| s as usize).product();
            let len = shape[axis] as usize;
            let inner: usize = shape[axis + 1..].iter().map(|&s| s as usize).product();
            let mut want = vec![0f64; outer * inner];
            for o in 0..outer {
                for j in 0..len {
                    for i in 0..inner {
                        want[o * inner + i] += xv[(o * len + j) * inner + i] as f64;
                    }
                }
            }
            let want: Vec<f32> = want.iter().map(|&v| v as f32).collect();
            let mut oshape = shape.clone();
            oshape.remove(axis);
            let (x, out) = (Buf::new(&xv), Buf::empty(outer * inner));
            let st = contiguous(&shape);
            assert_eq!(exec(op::SUM_AXIS, [axis as i64, 0, 0, 0], &[(&x, &shape, &st)], &out, &oshape), STATUS_OK);
            close(&format!("sum {shape:?} axis {axis}"), &out.get(), &want, 1e-4);
            // The result must not depend on scheduling: a second run is bit-identical.
            let out2 = Buf::empty(outer * inner);
            exec(op::SUM_AXIS, [axis as i64, 0, 0, 0], &[(&x, &shape, &st)], &out2, &oshape);
            assert_eq!(out.get().iter().map(|v| v.to_bits()).collect::<Vec<_>>(), out2.get().iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "sum {shape:?} axis {axis} not deterministic");
        }
    }
}

#[test]
fn vector_math_edge_values() {
    let mut vals: Vec<f32> = vec![
        0.0, -0.0, 1e-45, 1e-40, 1e-38, 1e-10, 0.1, 0.5, 1.0, 10.0, 88.0, 88.4, 88.5, 88.7, 89.0, 100.0, 1e30, f32::INFINITY, -1e-45, -1e-10, -0.5, -10.0, -87.0, -88.0, -100.0,
        -104.0, -1e30, f32::NEG_INFINITY, f32::NAN,
    ];
    vals.extend((-300..300).map(|i| i as f32 * 0.37));
    vals.extend((1..2000).map(|i| i as f32 * 1e-3));
    let n = vals.len();
    let x = Buf::new(&vals);
    for (code, name, f) in [
        (op::EXP, "exp", (|v: f64| v.exp()) as fn(f64) -> f64),
        (op::LOG, "log", |v: f64| v.ln()),
        (op::TANH, "tanh", |v: f64| v.tanh()),
    ] {
        let out = Buf::empty(n);
        let (sh, st) = ([n as u64], [1u64]);
        assert_eq!(exec(code, [0; 4], &[(&x, &sh, &st)], &out, &sh), STATUS_OK);
        for (i, (&g, &v)) in out.get().iter().zip(&vals).enumerate() {
            let want = f(v as f64);
            if want.is_nan() {
                assert!(g.is_nan(), "{name}({v}) = {g}, want NaN (index {i})");
            } else if want.is_infinite() || want > f32::MAX as f64 {
                assert_eq!(g, if want > 0.0 { f32::INFINITY } else { f32::NEG_INFINITY }, "{name}({v}) (index {i})");
            } else {
                let want = want as f32;
                // A few ulps, or an absolute floor where the result is denormal or the true value is ~0.
                assert!((g - want).abs() <= 4e-7 * want.abs() + 2e-45 + f32::MIN_POSITIVE * 1e-1 * (want.abs() < f32::MIN_POSITIVE) as i32 as f32, "{name}({v}) = {g}, want {want} (index {i})");
            }
        }
    }
}

#[test]
fn axpy_in_place_and_out_of_place() {
    for n in [1usize, 7, 8, 33, 4097, 300_000] {
        let (av, bv) = (data(n, 8), data(n, 9));
        let want: Vec<f32> = av.iter().zip(&bv).map(|(a, b)| a - 0.25 * b).collect();
        let (a, b) = (Buf::new(&av), Buf::new(&bv));
        let (sh, st) = ([n as u64], [1u64]);
        let alpha = (-0.25f32).to_bits() as i64;
        // In place: the output is the first input's buffer.
        assert_eq!(exec(op::AXPY, [alpha, 0, 0, 0], &[(&a, &sh, &st), (&b, &sh, &st)], &a, &sh), STATUS_OK);
        close(&format!("axpy in place n={n}"), &a.get(), &want, 1e-6);
        let (a2, out) = (Buf::new(&av), Buf::empty(n));
        assert_eq!(exec(op::AXPY, [alpha, 0, 0, 0], &[(&a2, &sh, &st), (&b, &sh, &st)], &out, &sh), STATUS_OK);
        close(&format!("axpy out of place n={n}"), &out.get(), &want, 1e-6);
    }
}

#[test]
fn transpose_copy_and_strided_unary() {
    for (r, c) in [(1usize, 9usize), (9, 1), (3, 5), (32, 32), (33, 31), (257, 129), (1000, 70)] {
        let xv = data(r * c, 10);
        let mut want = vec![0f32; r * c];
        for i in 0..r {
            for j in 0..c {
                want[j * r + i] = xv[i * c + j];
            }
        }
        let x = Buf::new(&xv);
        // The transposed operand: shape [c, r] with strides [1, c] over the [r, c] storage.
        let (sh, st) = ([c as u64, r as u64], [1u64, c as u64]);
        let out = Buf::empty(r * c);
        assert_eq!(exec(op::COPY, [0; 4], &[(&x, &sh, &st)], &out, &sh), STATUS_OK);
        assert_eq!(out.get(), want, "transpose {r}x{c}");
        // Unary on a strided view goes through the gather path.
        let out = Buf::empty(r * c);
        assert_eq!(exec(op::NEG, [0; 4], &[(&x, &sh, &st)], &out, &sh), STATUS_OK);
        assert_eq!(out.get(), want.iter().map(|v| -v).collect::<Vec<_>>(), "neg of transpose {r}x{c}");
    }
}

#[test]
fn freed_buffers_are_reused() {
    let mut a = null_mut();
    let bytes = 3 << 20;
    assert_eq!(unsafe { (vt().alloc)(0, bytes, &mut a) }, STATUS_OK);
    unsafe { (vt().free)(0, a) };
    let mut b = null_mut();
    assert_eq!(unsafe { (vt().alloc)(0, bytes, &mut b) }, STATUS_OK);
    assert_eq!(a, b, "same-size allocation did not reuse the freed buffer");
    unsafe { (vt().free)(0, b) };
}
