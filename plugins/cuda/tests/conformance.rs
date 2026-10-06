//! Compares the CUDA plugin against the CPU plugin through the core Tensor API.
//! Skips (passes) when no CUDA device is available. Both plugins are loaded as DLLs.

use pytorches_core::plugin::load_plugin_file;
use pytorches_core::{Device, Tensor};
use pytorches_plugin_abi as abi;
use std::path::PathBuf;
use std::sync::{Once, OnceLock};
use std::time::Instant;

fn bin_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../bin")
}

fn cuda_dll() -> PathBuf {
    bin_dir().join("pytorches_plugin_cuda.dll")
}

/// Loads both plugins once; returns (cpu, cuda) devices, or None when CUDA is unavailable.
fn devices() -> Option<(Device, Device)> {
    static INIT: Once = Once::new();
    static OK: OnceLock<bool> = OnceLock::new();
    INIT.call_once(|| {
        // Prefer the freshly built dll from the cargo target dir.
        let built = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/release/pytorches_plugin_cuda.dll");
        if built.exists() {
            let _ = std::fs::copy(&built, cuda_dll());
        }
        load_plugin_file(&bin_dir().join("pytorches_plugin_cpu.dll")).expect("cpu plugin");
        let ok = match load_plugin_file(&cuda_dll()) {
            Ok(_) => true,
            Err(e) => {
                eprintln!("SKIP: cuda plugin unavailable: {e}");
                false
            }
        };
        OK.set(ok).unwrap();
    });
    if !*OK.get().unwrap() {
        return None;
    }
    Some((Device::parse("cpu:0").unwrap(), Device::parse("cuda:0").ok()?))
}

macro_rules! need_cuda {
    () => {
        match devices() {
            Some(d) => d,
            None => {
                eprintln!("SKIP: no CUDA device");
                return;
            }
        }
    };
}

/// Deterministic pseudo-random data in [-1, 1).
fn data(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / 8388608.0) - 1.0
        })
        .collect()
}

fn assert_close(what: &str, got: &[f32], want: &[f32], rtol: f32) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let tol = rtol * w.abs().max(1.0);
        assert!((g - w).abs() <= tol || (g.is_nan() && w.is_nan()), "{what}[{i}]: cuda {g} vs cpu {w}");
    }
}

fn pair(shape: &[usize], seed: u64, cpu: &Device, gpu: &Device) -> (Tensor, Tensor) {
    let d = data(shape.iter().product(), seed);
    (Tensor::from_vec_on(d.clone(), shape.to_vec(), cpu), Tensor::from_vec_on(d, shape.to_vec(), gpu))
}

#[test]
fn broadcasting_binary_ops() {
    let (cpu, gpu) = need_cuda!();
    let cases: [(&[usize], &[usize]); 7] = [
        (&[2, 3], &[3]),
        (&[4, 1], &[1, 5]),
        (&[2, 3, 4], &[3, 1]),
        (&[2, 1, 4], &[1, 3, 1]),
        (&[1], &[5, 7]),
        (&[], &[3, 3]),
        (&[3, 33, 17], &[3, 33, 17]),
    ];
    for (i, (sa, sb)) in cases.iter().enumerate() {
        let (ac, ag) = pair(sa, 10 + i as u64, &cpu, &gpu);
        let (bc, bg) = pair(sb, 50 + i as u64, &cpu, &gpu);
        // Keep divisors away from zero.
        let (bc, bg) = (bc.add(&Tensor::scalar_on(2.0, &cpu)), bg.add(&Tensor::scalar_on(2.0, &gpu)));
        for (name, f) in [
            ("add", Tensor::add as fn(&Tensor, &Tensor) -> Tensor),
            ("sub", Tensor::sub),
            ("mul", Tensor::mul),
            ("div", Tensor::div),
        ] {
            let c = f(&ac, &bc);
            let g = f(&ag, &bg);
            assert_eq!(c.shape(), g.shape());
            assert_close(&format!("{name} {sa:?}{sb:?}"), &g.to_vec(), &c.to_vec(), 1e-6);
        }
    }
}

#[test]
fn transpose_strided_copy() {
    let (cpu, gpu) = need_cuda!();
    for shape in [[2usize, 3], [33, 65], [1, 7], [100, 1], [257, 129]] {
        let (c, g) = pair(&shape, 7, &cpu, &gpu);
        assert_eq!(g.t().to_vec(), c.t().to_vec(), "transpose {shape:?}");
        assert_eq!(g.t().t().to_vec(), c.to_vec());
    }
}

#[test]
fn unary_ops() {
    let (cpu, gpu) = need_cuda!();
    let (c, g) = pair(&[1000], 3, &cpu, &gpu);
    assert_close("neg", &g.neg().to_vec(), &c.neg().to_vec(), 0.0);
    assert_close("relu", &g.relu().to_vec(), &c.relu().to_vec(), 0.0);
    assert_close("tanh", &g.tanh().to_vec(), &c.tanh().to_vec(), 1e-5);
    assert_close("exp", &g.exp().to_vec(), &c.exp().to_vec(), 1e-5);
    let (pc, pg) = (c.mul(&c).add(&Tensor::scalar_on(0.01, &cpu)), g.mul(&g).add(&Tensor::scalar_on(0.01, &gpu)));
    assert_close("log", &pg.log().to_vec(), &pc.log().to_vec(), 1e-5);
}

#[test]
fn matmul_sizes() {
    let (cpu, gpu) = need_cuda!();
    for (m, k, n) in [(33, 65, 17), (1, 1, 1), (5, 129, 130), (128, 8, 128), (512, 512, 512), (7, 300, 1)] {
        let (ac, ag) = pair(&[m, k], 1, &cpu, &gpu);
        let (bc, bg) = pair(&[k, n], 2, &cpu, &gpu);
        assert_close(&format!("matmul {m}x{k}x{n}"), &ag.matmul(&bg).to_vec(), &ac.matmul(&bc).to_vec(), 2e-4);
    }
}

#[test]
fn sums_and_means() {
    let (cpu, gpu) = need_cuda!();
    // sum() reduces a flat tensor (inner == 1); big ones exercise the multi-block path.
    for n in [1usize, 7, 1000, 100_000, 3_000_001] {
        let (c, g) = pair(&[n], 4, &cpu, &gpu);
        assert_close(&format!("sum {n}"), &g.sum().to_vec(), &c.sum().to_vec(), 1e-3 + 1e-6 * n as f32);
        assert_close(&format!("mean {n}"), &g.mean().to_vec(), &c.mean().to_vec(), 1e-4);
    }
    // Axis sums happen through broadcast backward: inner > 1, small/large totals, 3-D.
    for (sa, sb) in [
        (vec![5000usize, 3], vec![3usize]),
        (vec![64, 40, 30], vec![40, 1]),
        (vec![300, 200], vec![1, 200]),
        (vec![2, 3, 4], vec![2, 1, 4]),
    ] {
        let (ac, ag) = pair(&sa, 5, &cpu, &gpu);
        let (bc, bg) = pair(&sb, 6, &cpu, &gpu);
        let (ac, bc) = (ac.requires_grad_(true), bc.requires_grad_(true));
        let (ag, bg) = (ag.requires_grad_(true), bg.requires_grad_(true));
        ac.add(&bc).sum().backward();
        ag.add(&bg).sum().backward();
        assert_close("grad b", &bg.grad().unwrap().to_vec(), &bc.grad().unwrap().to_vec(), 1e-5);
        assert_close("grad a", &ag.grad().unwrap().to_vec(), &ac.grad().unwrap().to_vec(), 0.0);
    }
}

#[test]
fn randn_fill_golden() {
    let (cpu, gpu) = need_cuda!();
    let v = Tensor::randn_on(&[4], 1234, &gpu).to_vec();
    for (got, want) in v.iter().zip([0.6574781f32, -0.08203645, -2.2065563, 0.70945626]) {
        assert!((got - want).abs() < 1e-4, "{v:?}");
    }
    let (g, c) = (Tensor::randn_on(&[100_000], 99, &gpu), Tensor::randn_on(&[100_000], 99, &cpu));
    assert_close("randn vs cpu", &g.to_vec(), &c.to_vec(), 1e-4);
    assert_eq!(Tensor::full_on(&[3, 5], 2.5, &gpu).to_vec(), vec![2.5; 15]);
    assert_eq!(Tensor::zeros_on(&[1], &gpu).to_vec(), vec![0.0]);
    let t = Tensor::zeros_on(&[2, 2], &gpu);
    t.copy_(&Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]));
    assert_eq!(t.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn mlp_gradients_match() {
    let (cpu, gpu) = need_cuda!();
    let run = |dev: &Device| {
        let mk = |shape: &[usize], seed| Tensor::from_vec_on(data(shape.iter().product(), seed), shape.to_vec(), dev);
        let x = mk(&[32, 10], 1);
        let t = mk(&[32, 3], 2);
        let w1 = mk(&[10, 20], 3).requires_grad_(true);
        let b1 = mk(&[20], 4).requires_grad_(true);
        let w2 = mk(&[20, 3], 5).requires_grad_(true);
        let diff = x.matmul(&w1).add(&b1).relu().matmul(&w2).sub(&t);
        let loss = diff.mul(&diff).mean();
        loss.backward();
        vec![loss.to_vec(), w1.grad().unwrap().to_vec(), b1.grad().unwrap().to_vec(), w2.grad().unwrap().to_vec()]
    };
    let (c, g) = (run(&cpu), run(&gpu));
    for (i, (gv, cv)) in g.iter().zip(&c).enumerate() {
        assert_close(&format!("mlp tensor {i}"), gv, cv, 1e-4);
    }
}

#[test]
fn allocation_one_gib() {
    let (_, gpu) = need_cuda!();
    let n = (1usize << 30) / 4;
    {
        let t = Tensor::full_on(&[n], 3.0, &gpu);
        let u = t.add(&Tensor::scalar_on(1.0, &gpu)); // another 1 GiB
        gpu.synchronize();
        let v = u.to_vec();
        assert!(v[0] == 4.0 && v[n / 2] == 4.0 && v[n - 1] == 4.0);
        assert_eq!(u.sum().item(), 4.0 * n as f32);
    }
    let info = gpu.info();
    eprintln!("{info:?}");
    assert_eq!(info.kind, abi::KIND_CUDA);
    assert!(info.total_memory.unwrap() > 1 << 30 && info.free_memory.unwrap() > 0);
}

#[test]
fn out_of_memory_is_a_status_not_a_crash() {
    let _ = need_cuda!();
    unsafe {
        let lib = libloading::Library::new(cuda_dll()).unwrap();
        let entry: abi::EntryFn = *lib.get::<abi::EntryFn>(abi::ENTRY_SYMBOL).unwrap();
        let vt = &*entry();
        let mut p = std::ptr::null_mut();
        let st = (vt.alloc)(0, 64usize << 30, &mut p);
        assert_eq!(st, abi::STATUS_OUT_OF_MEMORY);
        let msg = std::ffi::CStr::from_ptr((vt.last_error)()).to_string_lossy().into_owned();
        eprintln!("oom message: {msg}");
        assert!(msg.contains("OUT_OF_MEMORY"), "{msg}");
        // The device still works afterwards.
        let st = (vt.alloc)(0, 1 << 20, &mut p);
        assert_eq!(st, abi::STATUS_OK);
        (vt.free)(0, p);
    }
}

#[test]
fn benchmark() {
    let (_, gpu) = need_cuda!();
    let n = 4096;
    let a = Tensor::randn_on(&[n, n], 1, &gpu);
    let b = Tensor::randn_on(&[n, n], 2, &gpu);
    let _ = a.matmul(&b);
    gpu.synchronize();
    let iters = 5;
    let t0 = Instant::now();
    let mut c = a.matmul(&b);
    for _ in 1..iters {
        c = a.matmul(&b);
    }
    gpu.synchronize();
    let dt = t0.elapsed().as_secs_f64() / iters as f64;
    let flops = 2.0 * (n as f64).powi(3);
    eprintln!("BENCH matmul {n}^3: {:.2} ms, {:.0} GFLOPs (c[0]={})", dt * 1e3, flops / dt / 1e9, c.to_vec()[0]);

    let m = 64usize << 20; // 64M floats = 256 MiB per tensor
    let x = Tensor::randn_on(&[m], 3, &gpu);
    let y = Tensor::randn_on(&[m], 4, &gpu);
    let _ = x.add(&y);
    gpu.synchronize();
    let t0 = Instant::now();
    let mut z = x.add(&y);
    for _ in 1..10 {
        z = x.add(&y);
    }
    gpu.synchronize();
    let dt = t0.elapsed().as_secs_f64() / 10.0;
    eprintln!("BENCH add {m} f32: {:.2} ms, {:.0} GB/s (z[0]={})", dt * 1e3, 12.0 * m as f64 / dt / 1e9, z.to_vec()[0]);

    let t0 = Instant::now();
    let s = x.sum();
    gpu.synchronize();
    let dt = t0.elapsed().as_secs_f64();
    eprintln!("BENCH sum {m} f32: {:.2} ms, {:.0} GB/s (s={})", dt * 1e3, 4.0 * m as f64 / dt / 1e9, s.item());
}
