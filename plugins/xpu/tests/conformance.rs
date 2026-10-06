//! Conformance + benchmark tests: xpu:0 vs cpu:0 through the core Tensor API.
//! Both plugins are loaded dynamically from `plugins/bin`. Skips (passes) without an Intel OpenCL GPU.
//! Run with `--nocapture` to see benchmark numbers.

use pytorches_core::plugin::load_plugin_file;
use pytorches_core::{Device, Tensor};
use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::Instant;

fn bin(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../bin").join(name)
}

struct Devs {
    cpu: Device,
    xpu: Device,
}

fn devs() -> Option<&'static Devs> {
    static D: OnceLock<Option<Devs>> = OnceLock::new();
    D.get_or_init(|| {
        load_plugin_file(&bin("pytorches_plugin_cpu.dll")).expect("cpu plugin");
        match load_plugin_file(&bin("pytorches_plugin_xpu.dll")) {
            Ok(_) => Some(Devs { cpu: Device::parse("cpu:0").unwrap(), xpu: Device::parse("xpu:0").unwrap() }),
            Err(e) => {
                eprintln!("SKIP: no usable Intel OpenCL GPU ({e})");
                None
            }
        }
    })
    .as_ref()
}

/// Heavy / timing tests take this lock so they do not run concurrently with each other.
fn heavy() -> MutexGuard<'static, ()> {
    static L: Mutex<()> = Mutex::new(());
    L.lock().unwrap_or_else(|p| p.into_inner())
}

macro_rules! need {
    () => {
        match devs() {
            Some(d) => d,
            None => return,
        }
    };
}

fn lcg(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / 16777216.0) * 2.0 - 1.0
        })
        .collect()
}

fn assert_close(a: &[f32], b: &[f32], tol: f32, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let ok = (x - y).abs() <= tol * (1.0 + y.abs()) || (x.is_nan() && y.is_nan());
        assert!(ok, "{what}: element {i}: xpu {x} vs cpu {y}");
    }
}

/// Builds the same tensor on both devices.
fn both(d: &Devs, shape: &[usize], seed: u64) -> (Tensor, Tensor) {
    let n: usize = shape.iter().product();
    let v = lcg(n, seed);
    (Tensor::from_vec_on(v.clone(), shape.to_vec(), &d.cpu), Tensor::from_vec_on(v, shape.to_vec(), &d.xpu))
}

#[test]
fn device_info() {
    let d = need!();
    let i = d.xpu.info();
    eprintln!("xpu:0 = {i:?}");
    assert_eq!(i.kind, pytorches_plugin_abi::KIND_XPU);
    assert!(i.total_memory.unwrap() > (1 << 30));
}

#[test]
fn unary_ops() {
    let d = need!();
    let (c, x) = both(d, &[7, 13], 1);
    assert_close(&x.neg().to_vec(), &c.neg().to_vec(), 1e-6, "neg");
    assert_close(&x.exp().to_vec(), &c.exp().to_vec(), 1e-5, "exp");
    assert_close(&x.relu().to_vec(), &c.relu().to_vec(), 1e-6, "relu");
    assert_close(&x.tanh().to_vec(), &c.tanh().to_vec(), 1e-5, "tanh");
    let pos = |t: &Tensor| t.exp(); // strictly positive input for log
    assert_close(&pos(&x).log().to_vec(), &pos(&c).log().to_vec(), 1e-5, "log");
    // relu derivative (STEP) is exercised through backward below
}

#[test]
fn broadcasting_binary() {
    let d = need!();
    let shapes: &[(&[usize], &[usize])] = &[
        (&[4, 5], &[4, 5]),
        (&[4, 5], &[5]),
        (&[4, 1], &[1, 5]),
        (&[3, 4, 5], &[4, 5]),
        (&[3, 1, 5], &[1, 4, 1]),
        (&[2, 3, 4, 5], &[3, 1, 5]),
        (&[1], &[6, 7]),
        (&[], &[5, 3]),
        (&[1, 1, 1], &[2, 3, 4]),
        (&[17, 33], &[17, 1]),
    ];
    for (k, (sa, sb)) in shapes.iter().enumerate() {
        let (ca, xa) = both(d, sa, 10 + k as u64);
        let (cb, xb) = both(d, sb, 50 + k as u64);
        let cb2 = cb.exp(); // keep divisor away from 0
        let xb2 = xb.exp();
        for (name, rc, rx) in [
            ("add", ca.add(&cb), xa.add(&xb)),
            ("sub", ca.sub(&cb), xa.sub(&xb)),
            ("mul", ca.mul(&cb), xa.mul(&xb)),
            ("div", ca.div(&cb2), xa.div(&xb2)),
        ] {
            assert_eq!(rc.shape(), rx.shape());
            assert_close(&rx.to_vec(), &rc.to_vec(), 1e-5, &format!("{name} {sa:?} {sb:?}"));
        }
    }
}

#[test]
fn transpose_strided_copy() {
    let d = need!();
    for shape in [[3usize, 5], [64, 33], [1, 9], [100, 1]] {
        let (c, x) = both(d, &shape, 7);
        assert_close(&x.t().to_vec(), &c.t().to_vec(), 0.0, "t");
    }
    // transpose of a transpose and use inside arithmetic
    let (c, x) = both(d, &[12, 20], 8);
    assert_close(&x.t().t().to_vec(), &c.t().t().to_vec(), 0.0, "t.t");
    let (c2, x2) = both(d, &[20, 12], 9);
    assert_close(&x.t().add(&x2).to_vec(), &c.t().add(&c2).to_vec(), 1e-6, "t+x");
}

#[test]
fn matmul_sizes() {
    let d = need!();
    for (m, k, n) in [
        (33, 65, 17),
        (1, 1, 1),
        (64, 16, 64),
        (65, 17, 65),
        (5, 300, 7),
        (128, 128, 128),
        (512, 512, 512),
        (3, 1000, 130),
        // Sub-group fast path: smallest tile, a few tiles, a long K, and each dimension one off the
        // tile size (those must fall back to the general kernel).
        (16, 16, 32),
        (48, 48, 96),
        (32, 4096, 64),
        (256, 2048, 64),
        (16, 16, 31),
        (17, 16, 32),
        (16, 17, 32),
        (16, 16, 33),
    ] {
        let (ca, xa) = both(d, &[m, k], 3);
        let (cb, xb) = both(d, &[k, n], 4);
        assert_close(&xa.matmul(&xb).to_vec(), &ca.matmul(&cb).to_vec(), 2e-4, &format!("matmul {m}x{k}x{n}"));
    }
}

#[test]
fn axpy_in_place() {
    let d = need!();
    for n in [1usize, 255, 4097, 1 << 20] {
        let (ca, xa) = both(d, &[n], 5);
        let (cb, xb) = both(d, &[n], 6);
        ca.axpy_(-0.25, &cb);
        xa.axpy_(-0.25, &xb);
        assert_close(&xa.to_vec(), &ca.to_vec(), 1e-6, &format!("axpy n={n}"));
    }
}

#[test]
fn sum_and_mean() {
    let d = need!();
    for shape in [vec![1000], vec![33, 17], vec![5, 4, 3], vec![70000], vec![3, 100000], vec![100000, 3], vec![2, 600, 5], vec![1, 1]] {
        let (c, x) = both(d, &shape, 21);
        assert_close(&[x.sum().item()], &[c.sum().item()], 1e-3, &format!("sum {shape:?}"));
        assert_close(&[x.mean().item()], &[c.mean().item()], 1e-3, &format!("mean {shape:?}"));
    }
    // axis sums come from broadcasting backward: (x + bias).sum() grads
    let (cx, xx) = both(d, &[4, 700], 31);
    let (cb, xb) = both(d, &[700], 32);
    let (cb, xb) = (cb.requires_grad_(true), xb.requires_grad_(true));
    let (lc, lx) = (cx.add(&cb).sum(), xx.add(&xb).sum());
    lc.backward();
    lx.backward();
    assert_close(&xb.grad().unwrap().to_vec(), &cb.grad().unwrap().to_vec(), 1e-5, "bias grad (axis sum)");
}

#[test]
fn fill_and_randn() {
    let d = need!();
    let f = Tensor::full_on(&[3, 5], 2.5, &d.xpu).to_vec();
    assert!(f.iter().all(|&v| v == 2.5));
    assert_eq!(Tensor::zeros_on(&[10], &d.xpu).to_vec(), vec![0.0; 10]);
    assert_eq!(Tensor::ones_on(&[2, 2], &d.xpu).to_vec(), vec![1.0; 4]);
    assert_eq!(Tensor::scalar_on(-3.0, &d.xpu).to_vec(), vec![-3.0]);

    let r = Tensor::randn_on(&[4], 1234, &d.xpu).to_vec();
    let golden = [0.6574781f32, -0.08203645, -2.2065563, 0.70945626];
    for (a, b) in r.iter().zip(golden) {
        assert!((a - b).abs() < 1e-4, "randn golden: {r:?}");
    }
    let n = 100_000;
    assert_close(
        &Tensor::randn_on(&[n], 77, &d.xpu).to_vec(),
        &Tensor::randn_on(&[n], 77, &d.cpu).to_vec(),
        2e-3,
        "randn vs cpu",
    );
    // copy_ (d2d / COPY op) round trip
    let (_, a) = both(d, &[9, 9], 5);
    let b = Tensor::zeros_on(&[9, 9], &d.xpu);
    b.copy_(&a);
    assert_eq!(a.to_vec(), b.to_vec());
}

#[test]
fn cross_device_transfer() {
    let d = need!();
    let (c, _) = both(d, &[50, 50], 2);
    let x = c.to(&d.xpu);
    assert_eq!(x.device(), d.xpu);
    assert_eq!(x.to(&d.cpu).to_vec(), c.to_vec());
}

#[test]
fn mlp_gradients() {
    let d = need!();
    let mk = |dev: &Device| {
        let t = |shape: &[usize], seed: u64, scale: f32| {
            let v: Vec<f32> = lcg(shape.iter().product(), seed).into_iter().map(|x| x * scale).collect();
            Tensor::from_vec_on(v, shape.to_vec(), dev)
        };
        let x = t(&[19, 23], 1, 1.0);
        let y = t(&[19, 5], 2, 1.0);
        let w1 = t(&[23, 37], 3, 0.3).requires_grad_(true);
        let b1 = t(&[37], 4, 0.1).requires_grad_(true);
        let w2 = t(&[37, 5], 5, 0.3).requires_grad_(true);
        let h = x.matmul(&w1).add(&b1).relu();
        let diff = h.matmul(&w2).sub(&y);
        let loss = diff.mul(&diff).mean();
        loss.backward();
        (loss.item(), w1.grad().unwrap().to_vec(), b1.grad().unwrap().to_vec(), w2.grad().unwrap().to_vec())
    };
    let (lc, g1c, gbc, g2c) = mk(&d.cpu);
    let (lx, g1x, gbx, g2x) = mk(&d.xpu);
    assert_close(&[lx], &[lc], 1e-4, "loss");
    assert_close(&g1x, &g1c, 1e-3, "grad w1");
    assert_close(&gbx, &gbc, 1e-3, "grad b1");
    assert_close(&g2x, &g2c, 1e-3, "grad w2");
    assert!(g1x.iter().any(|v| v.abs() > 1e-6));
}

#[test]
#[ignore = "allocates many GiB; run with --ignored"]
fn large_allocation_4gib_and_more() {
    let d = need!();
    let _g = heavy();
    let n: usize = 1 << 30; // 4 GiB of f32
    let t = Instant::now();
    let a = Tensor::full_on(&[n], 1.5, &d.xpu);
    let s = a.sum().item();
    eprintln!("4 GiB tensor: full + sum = {s} in {:.2?}", t.elapsed());
    assert_eq!(s, 1.5 * n as f32);
    // 3 x 4 GiB = 12 GiB alive, binary op reads/writes beyond the 4 GiB mark
    let b = Tensor::full_on(&[n], 2.5, &d.xpu);
    let c = a.add(&b);
    assert_eq!(c.sum().item(), 4.0 * n as f32);
    drop((a, b, c));
    // a single 8 GiB tensor, if the driver allows it
    let big = Tensor::full_on(&[2 * n], 0.5, &d.xpu);
    assert_eq!(big.sum().item(), 0.5 * (2 * n) as f32);
    eprintln!("8 GiB single tensor OK; free estimate {:?}", d.xpu.info().free_memory);
}

#[test]
#[ignore = "allocates many GiB; run with --ignored"]
fn many_buffers_alive() {
    let d = need!();
    let _g = heavy();
    let n: usize = 1 << 28; // 1 GiB
    let ts: Vec<Tensor> = (0..12).map(|i| Tensor::full_on(&[n], i as f32, &d.xpu)).collect();
    for (i, t) in ts.iter().enumerate() {
        assert_eq!(t.sum().item(), i as f32 * n as f32, "buffer {i}");
    }
    let info = d.xpu.info();
    eprintln!("12 x 1 GiB alive; device free estimate = {} MiB of {} MiB", info.free_memory.unwrap() >> 20, info.total_memory.unwrap() >> 20);
    assert!(info.free_memory.unwrap() + (12u64 << 30) <= info.total_memory.unwrap());
    drop(ts);
    d.xpu.synchronize();
}

#[test]
fn oom_is_an_error_not_an_abort() {
    let d = need!();
    let _g = heavy();
    // Directly ask for far more than the machine has; the core would assert, so go through the vtable semantics
    // indirectly: allocations of the device total + 1 GiB must fail cleanly. We exercise this in tests/limits.rs.
    let _ = &d.xpu;
}

#[test]
#[ignore = "allocates many GiB; run with --ignored"]
fn benchmarks() {
    let d = need!();
    let _g = heavy();
    for &sz in &[2048usize, 4096] {
        let a = Tensor::randn_on(&[sz, sz], 1, &d.xpu);
        let b = Tensor::randn_on(&[sz, sz], 2, &d.xpu);
        let _ = a.matmul(&b);
        d.xpu.synchronize();
        let iters = 5;
        let t = Instant::now();
        let mut last = a.matmul(&b);
        for _ in 1..iters {
            last = a.matmul(&b);
        }
        d.xpu.synchronize();
        let dt = t.elapsed().as_secs_f64() / iters as f64;
        let flops = 2.0 * (sz as f64).powi(3);
        eprintln!("BENCH matmul {sz}x{sz}: {:.2} ms, {:.0} GFLOP/s", dt * 1e3, flops / dt / 1e9);
        drop(last);
    }
    let n = 1usize << 27; // 512 MiB per tensor
    let a = Tensor::full_on(&[n], 1.0, &d.xpu);
    let b = Tensor::full_on(&[n], 2.0, &d.xpu);
    let _ = a.add(&b);
    d.xpu.synchronize();
    let iters = 10;
    let t = Instant::now();
    let mut c = a.add(&b);
    for _ in 1..iters {
        c = a.add(&b);
    }
    d.xpu.synchronize();
    let dt = t.elapsed().as_secs_f64() / iters as f64;
    eprintln!("BENCH add 512 MiB x2 -> 512 MiB: {:.2} ms, {:.1} GB/s", dt * 1e3, 3.0 * (n * 4) as f64 / dt / 1e9);
    let t = Instant::now();
    for _ in 0..iters {
        c.copy_(&a);
    }
    d.xpu.synchronize();
    let dt = t.elapsed().as_secs_f64() / iters as f64;
    eprintln!("BENCH copy_ into existing 512 MiB (no alloc): {:.2} ms, {:.1} GB/s", dt * 1e3, 2.0 * (n * 4) as f64 / dt / 1e9);
    let t = Instant::now();
    let mut s = a.sum();
    for _ in 1..iters {
        s = a.sum();
    }
    d.xpu.synchronize();
    let dt = t.elapsed().as_secs_f64() / iters as f64;
    eprintln!("BENCH sum 512 MiB: {:.2} ms, {:.1} GB/s ({})", dt * 1e3, (n * 4) as f64 / dt / 1e9, s.item());
    let t = Instant::now();
    let mut e = a.exp();
    for _ in 1..iters {
        e = a.exp();
    }
    d.xpu.synchronize();
    let dt = t.elapsed().as_secs_f64() / iters as f64;
    eprintln!("BENCH exp 512 MiB -> 512 MiB: {:.2} ms, {:.1} GB/s", dt * 1e3, 2.0 * (n * 4) as f64 / dt / 1e9);
    drop((c, e));
}
