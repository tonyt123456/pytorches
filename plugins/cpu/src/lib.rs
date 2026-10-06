//! Reference CPU plugin. Correctness first: no SIMD, no threading.
//!
//! Exported as a shared library (`pytorches_plugin_cpu.dll`) and also as an rlib so the
//! core's own tests can register it statically.

use pytorches_plugin_abi::*;
use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::borrow::Cow;
use std::ffi::{c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::slice;

const HEADER: usize = 64; // also the allocation alignment
const ERR: &[u8] = b"cpu plugin: operation failed\0";

// ---- memory -----------------------------------------------------------------

unsafe extern "C" fn alloc_buf(_device: u32, bytes: usize, out: *mut *mut c_void) -> Status {
    let total = bytes + HEADER;
    let Ok(layout) = Layout::from_size_align(total, HEADER) else {
        return STATUS_INVALID_ARGUMENT;
    };
    unsafe {
        let base = alloc_zeroed(layout);
        if base.is_null() {
            return STATUS_OUT_OF_MEMORY;
        }
        *(base as *mut usize) = total;
        *out = base.add(HEADER) as *mut c_void;
    }
    STATUS_OK
}

unsafe extern "C" fn free_buf(_device: u32, ptr: *mut c_void) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let base = (ptr as *mut u8).sub(HEADER);
        let total = *(base as *const usize);
        dealloc(base, Layout::from_size_align_unchecked(total, HEADER));
    }
}

unsafe extern "C" fn copy_from_host(_d: u32, dst: *mut c_void, src: *const c_void, bytes: usize) -> Status {
    unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, bytes) };
    STATUS_OK
}

unsafe extern "C" fn copy_to_host(_d: u32, dst: *mut c_void, src: *const c_void, bytes: usize) -> Status {
    unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, bytes) };
    STATUS_OK
}

unsafe extern "C" fn copy_d2d(_d: u32, dst: *mut c_void, src: *const c_void, bytes: usize) -> Status {
    unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, bytes) };
    STATUS_OK
}

// ---- device info ------------------------------------------------------------

unsafe extern "C" fn device_count() -> u32 {
    1
}

/// (total, available) system memory in bytes, if the OS will tell us.
#[cfg(windows)]
fn host_memory() -> Option<(u64, u64)> {
    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(buf: *mut MemoryStatusEx) -> i32;
    }
    let mut m = MemoryStatusEx {
        length: std::mem::size_of::<MemoryStatusEx>() as u32,
        memory_load: 0,
        total_phys: 0,
        avail_phys: 0,
        total_page_file: 0,
        avail_page_file: 0,
        total_virtual: 0,
        avail_virtual: 0,
        avail_extended_virtual: 0,
    };
    (unsafe { GlobalMemoryStatusEx(&mut m) } != 0).then_some((m.total_phys, m.avail_phys))
}

#[cfg(target_os = "linux")]
fn host_memory() -> Option<(u64, u64)> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let kb = |key: &str| -> Option<u64> {
        text.lines().find(|l| l.starts_with(key))?.split_whitespace().nth(1)?.parse::<u64>().ok().map(|v| v * 1024)
    };
    Some((kb("MemTotal:")?, kb("MemAvailable:")?))
}

#[cfg(not(any(windows, target_os = "linux")))]
fn host_memory() -> Option<(u64, u64)> {
    None
}

unsafe extern "C" fn device_info(device: u32, out: *mut DeviceInfo) -> Status {
    if device != 0 || out.is_null() {
        return STATUS_INVALID_ARGUMENT;
    }
    let (total, free) = host_memory().unwrap_or((MEMORY_UNKNOWN, MEMORY_UNKNOWN));
    let mut info = DeviceInfo {
        name: [0; 64],
        kind: KIND_CPU,
        total_memory: total,
        free_memory: free,
    };
    for (dst, &b) in info.name.iter_mut().zip(b"cpu") {
        *dst = b as c_char;
    }
    unsafe { *out = info };
    STATUS_OK
}

unsafe extern "C" fn synchronize(_device: u32) -> Status {
    STATUS_OK
}

unsafe extern "C" fn last_error() -> *const c_char {
    ERR.as_ptr() as *const c_char
}

// ---- kernels ----------------------------------------------------------------

unsafe extern "C" fn supports_op(code: u32) -> u32 {
    use op::*;
    matches!(
        code,
        NEG | EXP | LOG | RELU | TANH | STEP | ADD | SUB | MUL | DIV | MATMUL | SUM_AXIS | COPY | FILL | RAND_NORMAL
    ) as u32
}

unsafe fn dims<'a>(d: &TensorDesc) -> (&'a [u64], &'a [u64]) {
    unsafe {
        (
            slice::from_raw_parts(d.shape, d.ndim as usize),
            slice::from_raw_parts(d.strides, d.ndim as usize),
        )
    }
}

fn numel(shape: &[u64]) -> usize {
    shape.iter().product::<u64>() as usize
}

fn is_contiguous(shape: &[u64], strides: &[u64]) -> bool {
    let mut expect = 1u64;
    for i in (0..shape.len()).rev() {
        if shape[i] != 1 && strides[i] != expect {
            return false;
        }
        expect *= shape[i];
    }
    true
}

/// Reads a tensor argument as a contiguous slice, gathering through its strides if needed.
unsafe fn read<'a>(d: &TensorDesc) -> Cow<'a, [f32]> {
    unsafe {
        let (shape, strides) = dims(d);
        let n = numel(shape);
        let ptr = d.data as *const f32;
        if is_contiguous(shape, strides) {
            return Cow::Borrowed(slice::from_raw_parts(ptr, n));
        }
        let mut out = Vec::with_capacity(n);
        let mut idx = vec![0u64; shape.len()];
        let mut off = 0usize;
        for _ in 0..n {
            out.push(*ptr.add(off));
            for dim in (0..shape.len()).rev() {
                idx[dim] += 1;
                off += strides[dim] as usize;
                if idx[dim] < shape[dim] {
                    break;
                }
                off -= (strides[dim] * shape[dim]) as usize;
                idx[dim] = 0;
            }
        }
        Cow::Owned(out)
    }
}

unsafe fn write<'a>(d: &TensorDesc) -> &'a mut [f32] {
    unsafe {
        let (shape, _) = dims(d);
        slice::from_raw_parts_mut(d.data as *mut f32, numel(shape))
    }
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

unsafe fn run(op_code: u32, attrs: &OpAttrs, ins: &[TensorDesc], outs: &[TensorDesc]) -> Status {
    use op::*;
    if outs.len() != 1 {
        return STATUS_INVALID_ARGUMENT;
    }
    unsafe {
        let out = write(&outs[0]);
        match op_code {
            FILL => out.fill(f32::from_bits(attrs.ints[0] as u32)),
            RAND_NORMAL => {
                let seed = attrs.ints[0] as u64;
                for (i, o) in out.iter_mut().enumerate() {
                    *o = rand_normal(seed, i as u64);
                }
            }
            NEG | EXP | LOG | RELU | TANH | STEP | COPY => {
                let x = read(&ins[0]);
                let f: fn(f32) -> f32 = match op_code {
                    NEG => |v| -v,
                    EXP => f32::exp,
                    LOG => f32::ln,
                    RELU => |v| v.max(0.0),
                    TANH => f32::tanh,
                    STEP => |v| if v > 0.0 { 1.0 } else { 0.0 },
                    _ => |v| v, // COPY
                };
                for (o, &v) in out.iter_mut().zip(x.iter()) {
                    *o = f(v);
                }
            }
            ADD | SUB | MUL | DIV => {
                let (a, b) = (read(&ins[0]), read(&ins[1]));
                let f: fn(f32, f32) -> f32 = match op_code {
                    ADD => |x, y| x + y,
                    SUB => |x, y| x - y,
                    MUL => |x, y| x * y,
                    _ => |x, y| x / y,
                };
                for ((o, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
                    *o = f(x, y);
                }
            }
            MATMUL => {
                let (sa, _) = dims(&ins[0]);
                let (sb, _) = dims(&ins[1]);
                let (m, k, n) = (sa[0] as usize, sa[1] as usize, sb[1] as usize);
                let (a, b) = (read(&ins[0]), read(&ins[1]));
                out.fill(0.0);
                for i in 0..m {
                    for p in 0..k {
                        let av = a[i * k + p];
                        let (brow, orow) = (&b[p * n..(p + 1) * n], &mut out[i * n..(i + 1) * n]);
                        for j in 0..n {
                            orow[j] += av * brow[j];
                        }
                    }
                }
            }
            SUM_AXIS => {
                let (shape, _) = dims(&ins[0]);
                let axis = attrs.ints[0] as usize;
                if axis >= shape.len() {
                    return STATUS_INVALID_ARGUMENT;
                }
                let x = read(&ins[0]);
                let outer = numel(&shape[..axis]);
                let n = shape[axis] as usize;
                let inner = numel(&shape[axis + 1..]);
                out.fill(0.0);
                for o in 0..outer {
                    for j in 0..n {
                        let src = &x[(o * n + j) * inner..(o * n + j + 1) * inner];
                        let dst = &mut out[o * inner..(o + 1) * inner];
                        for i in 0..inner {
                            dst[i] += src[i];
                        }
                    }
                }
            }
            _ => return STATUS_UNSUPPORTED,
        }
    }
    STATUS_OK
}

unsafe extern "C" fn execute(
    _device: u32,
    op_code: u32,
    attrs: *const OpAttrs,
    inputs: *const TensorDesc,
    n_inputs: u32,
    outputs: *const TensorDesc,
    n_outputs: u32,
) -> Status {
    // Never unwind across the C boundary.
    catch_unwind(AssertUnwindSafe(|| unsafe {
        let attrs = if attrs.is_null() { OpAttrs::default() } else { *attrs };
        let ins = slice::from_raw_parts(inputs, n_inputs as usize);
        let outs = slice::from_raw_parts(outputs, n_outputs as usize);
        run(op_code, &attrs, ins, outs)
    }))
    .unwrap_or(STATUS_INTERNAL)
}

// ---- entry point --------------------------------------------------------------

static VTABLE: PluginVTable = PluginVTable {
    abi_version: ABI_VERSION,
    plugin_version: 1,
    name: c"cpu".as_ptr(),
    device_count,
    device_info,
    alloc: alloc_buf,
    free: free_buf,
    copy_from_host,
    copy_to_host,
    copy_device_to_device: copy_d2d,
    supports_op,
    execute,
    synchronize,
    last_error,
};

#[unsafe(no_mangle)]
pub extern "C" fn pytorches_plugin_entry() -> *const PluginVTable {
    &VTABLE
}
