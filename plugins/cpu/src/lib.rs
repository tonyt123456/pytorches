//! CPU plugin: multi-threaded and vectorized.
//!
//! * Elementwise ops walk broadcast/strided operands in place, in parallel, with AVX2/FMA inner loops
//!   (`ew`); `exp`, `log` and `tanh` use vector polynomial versions (`simd`).
//! * Matmul is a packed, cache-blocked GEMM with a 6x16 AVX2/FMA microkernel and arbitrary operand
//!   strides, so transposes are free (`gemm`).
//! * Sums are blocked and combined in a fixed order, so results do not depend on the thread count
//!   (`reduce`).
//! * Freed buffers are cached by size and reused, up to a cap, so a hot loop does not page-fault fresh
//!   memory for every op. Buffer contents are undefined after `alloc`.
//!
//! `PYTORCHES_CPU_THREADS` sets the number of worker threads (default: all logical cores).
//! Exported as a shared library (`pytorches_plugin_cpu.dll`) and also as an rlib so the core's own
//! tests can register it statically.

mod ew;
mod gemm;
mod pool;
mod reduce;
mod simd;

use pytorches_plugin_abi::*;
use std::alloc::{Layout, alloc, dealloc};
use std::collections::HashMap;
use std::ffi::{c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::slice;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

const HEADER: usize = 64; // also the allocation alignment
const ERR: &[u8] = b"cpu plugin: operation failed\0";

// ---- memory -----------------------------------------------------------------

/// Rounds a request up so similar sizes share buffers: powers of two below 1 MiB, 2 MiB steps above.
fn bucket(bytes: usize) -> usize {
    const MIB: usize = 1 << 20;
    if bytes < MIB { bytes.max(64).next_power_of_two() } else { bytes.div_ceil(2 * MIB) * 2 * MIB }
}

#[derive(Default)]
struct Cache {
    /// Free buffers by (bucketed) payload size; pointers are stored as `usize`.
    free: HashMap<usize, Vec<usize>>,
}

static CACHE: Mutex<Option<Cache>> = Mutex::new(None);
static CACHED_BYTES: AtomicUsize = AtomicUsize::new(0);

/// Most memory the cache may hold: an eighth of physical memory, at most 4 GiB.
fn cache_cap() -> usize {
    static CAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        let total = host_memory().map(|(t, _)| t as usize).unwrap_or(8 << 30);
        (total / 8).min(4 << 30)
    })
}

fn lock_cache() -> std::sync::MutexGuard<'static, Option<Cache>> {
    let mut g = CACHE.lock().unwrap_or_else(|p| p.into_inner());
    g.get_or_insert_with(Cache::default);
    g
}

unsafe fn release_block(payload: usize, size: usize) {
    unsafe {
        let base = (payload as *mut u8).sub(HEADER);
        dealloc(base, Layout::from_size_align_unchecked(size + HEADER, HEADER));
    }
}

fn flush_cache(c: &mut Cache) {
    for (size, ptrs) in c.free.drain() {
        for p in ptrs {
            unsafe { release_block(p, size) };
            CACHED_BYTES.fetch_sub(size, Ordering::Relaxed);
        }
    }
}

unsafe fn fresh_block(size: usize) -> *mut u8 {
    unsafe {
        let Ok(layout) = Layout::from_size_align(size + HEADER, HEADER) else { return std::ptr::null_mut() };
        let base = alloc(layout);
        if base.is_null() {
            return base;
        }
        *(base as *mut usize) = size;
        base.add(HEADER)
    }
}

unsafe extern "C" fn alloc_buf(_device: u32, bytes: usize, out: *mut *mut c_void) -> Status {
    let size = bucket(bytes);
    if size < bytes || size > isize::MAX as usize - HEADER {
        return STATUS_OUT_OF_MEMORY;
    }
    let mut g = lock_cache();
    let c = g.as_mut().unwrap();
    unsafe {
        if let Some(p) = c.free.get_mut(&size).and_then(|v| v.pop()) {
            CACHED_BYTES.fetch_sub(size, Ordering::Relaxed);
            *out = p as *mut c_void;
            return STATUS_OK;
        }
        let mut p = fresh_block(size);
        if p.is_null() {
            // Give the cache back to the OS and try once more before reporting out-of-memory.
            flush_cache(c);
            p = fresh_block(size);
            if p.is_null() {
                return STATUS_OUT_OF_MEMORY;
            }
        }
        *out = p as *mut c_void;
    }
    STATUS_OK
}

unsafe extern "C" fn free_buf(_device: u32, ptr: *mut c_void) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let size = *((ptr as *mut u8).sub(HEADER) as *const usize);
        if CACHED_BYTES.load(Ordering::Relaxed) + size > cache_cap() {
            return release_block(ptr as usize, size);
        }
        let mut g = lock_cache();
        g.as_mut().unwrap().free.entry(size).or_default().push(ptr as usize);
        CACHED_BYTES.fetch_add(size, Ordering::Relaxed);
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
    // Cached buffers are reusable, so they count as free.
    let free = if free == MEMORY_UNKNOWN { free } else { free.saturating_add(CACHED_BYTES.load(Ordering::Relaxed) as u64).min(total) };
    let mut info = DeviceInfo { name: [0; 64], kind: KIND_CPU, total_memory: total, free_memory: free, flags: 0 };
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
        NEG | EXP | LOG | RELU | TANH | STEP | ADD | SUB | MUL | DIV | MATMUL | MATMUL_T | AXPY | SUM_AXIS | COPY | FILL | RAND_NORMAL
    ) as u32
}

unsafe fn dims<'a>(d: &TensorDesc) -> (&'a [u64], &'a [u64]) {
    unsafe {
        if d.ndim == 0 {
            (&[], &[])
        } else {
            (slice::from_raw_parts(d.shape, d.ndim as usize), slice::from_raw_parts(d.strides, d.ndim as usize))
        }
    }
}

fn numel(shape: &[u64]) -> usize {
    shape.iter().product::<u64>() as usize
}

unsafe fn run(op_code: u32, attrs: &OpAttrs, ins: &[TensorDesc], outs: &[TensorDesc]) -> Status {
    use op::*;
    if outs.len() != 1 {
        return STATUS_INVALID_ARGUMENT;
    }
    let need = match op_code {
        FILL | RAND_NORMAL => 0,
        NEG | EXP | LOG | RELU | TANH | STEP | COPY | SUM_AXIS => 1,
        ADD | SUB | MUL | DIV | MATMUL | MATMUL_T | AXPY => 2,
        _ => return STATUS_UNSUPPORTED,
    };
    if ins.len() != need {
        return STATUS_INVALID_ARGUMENT;
    }
    unsafe {
        let (oshape, _) = dims(&outs[0]);
        let n = numel(oshape);
        let out = outs[0].data as *mut f32;
        match op_code {
            FILL => ew::fill(out, n, f32::from_bits(attrs.ints[0] as u32)),
            RAND_NORMAL => ew::randn(out, n, attrs.ints[0] as u64),
            NEG | EXP | LOG | RELU | TANH | STEP | COPY => {
                let (xs, xst) = dims(&ins[0]);
                if numel(xs) != n {
                    return STATUS_INVALID_ARGUMENT;
                }
                ew::unary(op_code, ins[0].data as *const f32, xs, xst, out);
            }
            ADD | SUB | MUL | DIV => {
                let ((sa, sta), (sb, stb)) = (dims(&ins[0]), dims(&ins[1]));
                if sa != oshape || sb != oshape {
                    return STATUS_INVALID_ARGUMENT;
                }
                ew::binary(op_code, ins[0].data as *const f32, sta, ins[1].data as *const f32, stb, oshape, out);
            }
            AXPY => {
                if numel(dims(&ins[0]).0) != n || numel(dims(&ins[1]).0) != n {
                    return STATUS_INVALID_ARGUMENT;
                }
                ew::axpy(ins[0].data as *const f32, ins[1].data as *const f32, out, f32::from_bits(attrs.ints[0] as u32), n);
            }
            MATMUL | MATMUL_T => {
                let ((sa, sta), (sb, stb)) = (dims(&ins[0]), dims(&ins[1]));
                if sa.len() != 2 || sb.len() != 2 || oshape.len() != 2 {
                    return STATUS_INVALID_ARGUMENT;
                }
                let flags = if op_code == MATMUL_T { attrs.ints[0] } else { 0 };
                let (ta, tb) = (flags & 1 != 0, flags & 2 != 0);
                // Logical A is [m,k], B is [k,n]; a flagged operand is the transpose of what is stored.
                let (m, k) = if ta { (sa[1], sa[0]) } else { (sa[0], sa[1]) };
                let (kb, nn) = if tb { (sb[1], sb[0]) } else { (sb[0], sb[1]) };
                if k != kb || oshape[0] != m || oshape[1] != nn {
                    return STATUS_INVALID_ARGUMENT;
                }
                let a = if ta {
                    gemm::Mat { ptr: ins[0].data as *const f32, rs: sta[1] as usize, cs: sta[0] as usize }
                } else {
                    gemm::Mat { ptr: ins[0].data as *const f32, rs: sta[0] as usize, cs: sta[1] as usize }
                };
                let b = if tb {
                    gemm::Mat { ptr: ins[1].data as *const f32, rs: stb[1] as usize, cs: stb[0] as usize }
                } else {
                    gemm::Mat { ptr: ins[1].data as *const f32, rs: stb[0] as usize, cs: stb[1] as usize }
                };
                gemm::sgemm(m as usize, nn as usize, k as usize, a, b, out);
            }
            SUM_AXIS => {
                let (shape, _) = dims(&ins[0]);
                let axis = attrs.ints[0] as usize;
                if attrs.ints[0] < 0 || axis >= shape.len() {
                    return STATUS_INVALID_ARGUMENT;
                }
                let outer = numel(&shape[..axis]);
                let len = shape[axis] as usize;
                let inner = numel(&shape[axis + 1..]);
                if outer * inner != n {
                    return STATUS_INVALID_ARGUMENT;
                }
                reduce::sum_axis(ins[0].data as *const f32, outer, len, inner, out);
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
        let ins: &[TensorDesc] = if n_inputs == 0 { &[] } else { slice::from_raw_parts(inputs, n_inputs as usize) };
        let outs: &[TensorDesc] = if n_outputs == 0 { &[] } else { slice::from_raw_parts(outputs, n_outputs as usize) };
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
