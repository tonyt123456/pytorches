//! Raw-ABI probe of the xpu plugin's large-allocation behaviour. Skips without an Intel OpenCL GPU.
//! Loads the DLL directly so it can read single elements at huge offsets through strided COPY.

use libloading::Library;
use pytorches_plugin_abi::*;
use std::ffi::c_void;
use std::path::PathBuf;
use std::ptr::null_mut;

fn dll() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../bin/pytorches_plugin_xpu.dll")
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}
fn rand_normal(seed: u64, i: u64) -> f32 {
    let h = splitmix64(seed ^ i.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let u1 = ((h >> 40) + 1) as f32 / 16777216.0;
    let u2 = (h & 0xFF_FFFF) as f32 / 16777216.0;
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

/// Reads `idx.len()` single elements at the given element offsets via a strided COPY (batch of single reads).
unsafe fn read_at(vt: &PluginVTable, buf: *mut c_void, idx: &[u64]) -> Vec<f32> {
    unsafe {
        let mut out = Vec::new();
        for &i in idx {
            // 1 element, but data pointer is the base: use shape [1,1], offset via a 2-D trick:
            // shape [2], stride [i] reads elements 0 and i.
            let shape = [2u64];
            let strides = [i];
            let inp = TensorDesc { data: buf, dtype: DTYPE_F32, ndim: 1, shape: shape.as_ptr(), strides: strides.as_ptr() };
            let mut dst = null_mut();
            let s = (vt.alloc)(0, 8, &mut dst); assert_eq!(s, STATUS_OK, "{}", std::ffi::CStr::from_ptr((vt.last_error)()).to_string_lossy());
            let ostr = [1u64];
            let o = TensorDesc { data: dst, dtype: DTYPE_F32, ndim: 1, shape: shape.as_ptr(), strides: ostr.as_ptr() };
            let st = (vt.execute)(0, op::COPY, std::ptr::null(), &inp, 1, &o, 1);
            assert_eq!(st, STATUS_OK, "strided read failed");
            let mut host = [0f32; 2];
            let s = (vt.copy_to_host)(0, host.as_mut_ptr() as *mut c_void, dst, 8); assert_eq!(s, STATUS_OK, "{}", std::ffi::CStr::from_ptr((vt.last_error)()).to_string_lossy());
            (vt.free)(0, dst);
            out.push(host[1]);
        }
        out
    }
}

#[test]
#[ignore = "allocates many GiB; run with --ignored"]
fn allocation_limits() {
    unsafe {
        std::env::set_var("PYTORCHES_XPU_VERBOSE", "1");
        let Ok(lib) = Library::new(dll()) else {
            eprintln!("SKIP: xpu dll not built");
            return;
        };
        let entry: unsafe extern "C" fn() -> *const PluginVTable = *lib.get(ENTRY_SYMBOL).unwrap();
        let vt = &*entry();
        if (vt.device_count)() == 0 {
            eprintln!("SKIP: no Intel OpenCL GPU");
            return;
        }
        let mut info = DeviceInfo { name: [0; 64], kind: 0, total_memory: 0, free_memory: 0 };
        assert_eq!((vt.device_info)(0, &mut info), STATUS_OK);
        let name = std::ffi::CStr::from_ptr(info.name.as_ptr()).to_string_lossy().into_owned();
        eprintln!("device '{name}' total={} MiB free={} MiB", info.total_memory >> 20, info.free_memory >> 20);

        let gib = 1u64 << 30;
        let fill_rand = |buf: *mut c_void, n: u64, seed: i64| {
            let shape = [n];
            let strides = [1u64];
            let o = TensorDesc { data: buf, dtype: DTYPE_F32, ndim: 1, shape: shape.as_ptr(), strides: strides.as_ptr() };
            let a = OpAttrs { ints: [seed, 0, 0, 0] };
            (vt.execute)(0, op::RAND_NORMAL, &a, std::ptr::null(), 0, &o, 1)
        };

        // Single big buffers: 4 GiB, then increasing sizes until the driver refuses.
        let list: Vec<u64> = std::env::var("LIMIT_GB").map(|s| s.split(',').map(|x| x.parse().unwrap()).collect()).unwrap_or(vec![4, 8, 16]);
        for gb in list {
            let bytes = (gb * gib) as usize;
            let mut p = null_mut();
            let st = (vt.alloc)(0, bytes, &mut p);
            if st != STATUS_OK {
                let msg = std::ffi::CStr::from_ptr((vt.last_error)()).to_string_lossy().into_owned();
                eprintln!("single buffer {gb} GiB: FAILED status {st}: {msg}");
                break;
            }
            let n = bytes as u64 / 4;
            let t = std::time::Instant::now();
            let st = fill_rand(p, n, 99);
            assert_eq!(st, STATUS_OK);
            assert_eq!((vt.synchronize)(0), STATUS_OK);
            let idx: Vec<u64> = [0, 12345, n / 3, n / 2 + 7, (1u64 << 30) + 3, n - 1].into_iter().filter(|&i| i < n).collect();
            let got = read_at(vt, p, &idx);
            for (i, g) in idx.iter().zip(&got) {
                let want = rand_normal(99, *i);
                assert!((g - want).abs() < 1e-3, "buffer {gb} GiB element {i}: got {g}, want {want}");
            }
            eprintln!("single buffer {gb} GiB: OK (rand fill + readback of tail elements, {:.2?})", t.elapsed());
            (vt.free)(0, p);
        }

        // Many buffers alive at once.
        let mut bufs = Vec::new();
        let mut total = 0u64;
        for i in 0..20 {
            let mut p = null_mut();
            let st = (vt.alloc)(0, gib as usize, &mut p);
            if st != STATUS_OK {
                eprintln!("1 GiB buffer #{i} failed (status {st}) after {total} GiB held");
                break;
            }
            assert_eq!(fill_rand(p, gib / 4, i as i64), STATUS_OK);
            bufs.push(p);
            total += 1;
        }
        assert_eq!((vt.synchronize)(0), STATUS_OK);
        for (i, &p) in bufs.iter().enumerate().step_by(7) {
            let got = read_at(vt, p, &[gib / 4 - 1]);
            let want = rand_normal(i as u64, gib / 4 - 1);
            assert!((got[0] - want).abs() < 1e-3);
        }
        assert_eq!((vt.device_info)(0, &mut info), STATUS_OK);
        eprintln!("held {total} x 1 GiB simultaneously; reported free={} MiB", info.free_memory >> 20);
        for p in bufs {
            (vt.free)(0, p);
        }
        assert!(total >= 16, "could only hold {total} GiB");
    }
}

#[test]
fn absurd_allocation_fails_cleanly() {
    unsafe {
        let Ok(lib) = Library::new(dll()) else { return };
        let entry: unsafe extern "C" fn() -> *const PluginVTable = *lib.get(ENTRY_SYMBOL).unwrap();
        let vt = &*entry();
        if (vt.device_count)() == 0 {
            return;
        }
        for bytes in [1usize << 40, 1usize << 50, usize::MAX / 2] {
            let mut p = null_mut();
            let st = (vt.alloc)(0, bytes, &mut p);
            let msg = std::ffi::CStr::from_ptr((vt.last_error)()).to_string_lossy().into_owned();
            eprintln!("alloc({bytes}) -> status {st}: {msg}");
            assert_eq!(st, STATUS_OUT_OF_MEMORY);
        }
    }
}
