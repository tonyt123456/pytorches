//! Intel GPU ("xpu") plugin for PyTorches, built on the OpenCL runtime that ships with the
//! Intel graphics driver (`OpenCL.dll`, loaded dynamically: no SDK or link-time dependency).
//!
//! * Only GPU devices with `CL_DEVICE_VENDOR_ID == 0x8086` are claimed.
//! * One context + one in-order command queue per device is the implicit default stream.
//! * Kernels (`kernels/kernels.cl`) are JIT-compiled on the first `execute` per device, with
//!   `-cl-intel-greater-than-4GB-buffer-required` so buffers/tensors beyond 4 GiB work.
//!   Buffers above 4 GiB (or `CL_DEVICE_MAX_MEM_ALLOC_SIZE`) are created with
//!   `CL_MEM_ALLOW_UNRESTRICTED_SIZE_INTEL`.
//! * The `cl_mem` handle itself is the opaque device pointer handed to the core.
//!
//! * Freed buffers go to a per-device cache keyed by bucketed size and are reused by later allocations
//!   (creating and committing a fresh multi-hundred-MiB buffer for every op was the main cost of
//!   elementwise ops). The cache is flushed when an allocation fails, so it never turns a request
//!   that would fit into an out-of-memory error.
//!
//! Environment knobs (all optional): `PYTORCHES_XPU_VERBOSE=1` prints device limits to stderr,
//! `PYTORCHES_XPU_ALLOC=host` adds `CL_MEM_ALLOC_HOST_PTR` to every buffer,
//! `PYTORCHES_XPU_BUILD_OPTS` replaces the kernel build options.

#![allow(non_snake_case)]
use libloading::Library;
use pytorches_plugin_abi::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr::{null, null_mut};
use std::slice;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

type H = *mut c_void;

const KERNEL_SRC: &str = include_str!("../kernels/kernels.cl");
const DEFAULT_BUILD_OPTS: &str = "-cl-intel-greater-than-4GB-buffer-required";
const INTEL_VENDOR_ID: u64 = 0x8086;

const CL_DEVICE_TYPE_GPU: u64 = 1 << 2;
const CL_DEVICE_VENDOR_ID: u32 = 0x1001;
const CL_DEVICE_MAX_MEM_ALLOC_SIZE: u32 = 0x1010;
const CL_DEVICE_GLOBAL_MEM_SIZE: u32 = 0x101F;
const CL_DEVICE_HOST_UNIFIED_MEMORY: u32 = 0x1035;
const CL_DEVICE_NAME: u32 = 0x102B;
const CL_PLATFORM_NAME: u32 = 0x0902;
const CL_PROGRAM_BUILD_LOG: u32 = 0x1183;
const CL_MEM_READ_WRITE: u64 = 1;
const CL_MEM_ALLOC_HOST_PTR: u64 = 1 << 4;
const CL_MEM_ALLOW_UNRESTRICTED_SIZE_INTEL: u64 = 1 << 23;

const FOUR_GIB: u64 = 1 << 32;
/// Host<->device transfers are split into pieces of this size.
const XFER_CHUNK: usize = 1 << 30;
/// Floats of partial-sum scratch per device (see `sum_axis`).
const SCRATCH_FLOATS: usize = 1 << 17;

// ---- error handling ---------------------------------------------------------------

struct XErr {
    status: Status,
    msg: String,
}

type R<T> = Result<T, XErr>;

fn fail<T>(status: Status, msg: impl Into<String>) -> R<T> {
    Err(XErr { status, msg: msg.into() })
}

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::new("no error").unwrap());
}

fn set_last_error(msg: &str) {
    let clean: String = msg.chars().filter(|&c| c != '\0').collect();
    LAST_ERROR.with(|e| *e.borrow_mut() = CString::new(format!("xpu plugin: {clean}")).unwrap());
}

fn cl_err_name(code: i32) -> &'static str {
    match code {
        0 => "CL_SUCCESS",
        -1 => "CL_DEVICE_NOT_FOUND",
        -2 => "CL_DEVICE_NOT_AVAILABLE",
        -3 => "CL_COMPILER_NOT_AVAILABLE",
        -4 => "CL_MEM_OBJECT_ALLOCATION_FAILURE",
        -5 => "CL_OUT_OF_RESOURCES",
        -6 => "CL_OUT_OF_HOST_MEMORY",
        -9 => "CL_MEM_COPY_OVERLAP/CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST",
        -11 => "CL_BUILD_PROGRAM_FAILURE",
        -30 => "CL_INVALID_VALUE",
        -33 => "CL_INVALID_DEVICE",
        -34 => "CL_INVALID_CONTEXT",
        -36 => "CL_INVALID_COMMAND_QUEUE",
        -37 => "CL_INVALID_HOST_PTR",
        -38 => "CL_INVALID_MEM_OBJECT",
        -43 => "CL_INVALID_BUILD_OPTIONS",
        -44 => "CL_INVALID_PROGRAM",
        -45 => "CL_INVALID_PROGRAM_EXECUTABLE",
        -46 => "CL_INVALID_KERNEL_NAME",
        -48 => "CL_INVALID_KERNEL",
        -49 => "CL_INVALID_ARG_INDEX",
        -50 => "CL_INVALID_ARG_VALUE",
        -51 => "CL_INVALID_ARG_SIZE",
        -52 => "CL_INVALID_KERNEL_ARGS",
        -53 => "CL_INVALID_WORK_DIMENSION",
        -54 => "CL_INVALID_WORK_GROUP_SIZE",
        -55 => "CL_INVALID_WORK_ITEM_SIZE",
        -59 => "CL_INVALID_OPERATION/CL_INVALID_GLOBAL_WORK_SIZE",
        -61 => "CL_INVALID_BUFFER_SIZE",
        _ => "CL_UNKNOWN_ERROR",
    }
}

fn is_oom_code(code: i32) -> bool {
    matches!(code, -4 | -5 | -6)
}

fn check(code: i32, what: &str) -> R<()> {
    if code == 0 {
        return Ok(());
    }
    let status = if is_oom_code(code) { STATUS_OUT_OF_MEMORY } else { STATUS_INTERNAL };
    fail(status, format!("{what}: {} ({code})", cl_err_name(code)))
}

fn guard(f: impl FnOnce() -> R<()>) -> Status {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => STATUS_OK,
        Ok(Err(e)) => {
            set_last_error(&e.msg);
            e.status
        }
        Err(_) => {
            set_last_error("internal panic");
            STATUS_INTERNAL
        }
    }
}

// ---- dynamic OpenCL ---------------------------------------------------------------

macro_rules! cl_api {
    ($($name:ident : $ty:ty,)*) => {
        #[allow(non_snake_case)]
        struct Cl {
            _lib: Library,
            $($name: $ty,)*
        }
        impl Cl {
            unsafe fn load() -> Result<Cl, String> {
                let lib = unsafe { Library::new(LIB_NAME) }.map_err(|e| e.to_string())?;
                $(
                    let $name: $ty = unsafe {
                        *lib.get::<$ty>(concat!(stringify!($name), "\0").as_bytes()).map_err(|e| e.to_string())?
                    };
                )*
                Ok(Cl { _lib: lib, $($name,)* })
            }
        }
    };
}

#[cfg(windows)]
const LIB_NAME: &str = "OpenCL.dll";
#[cfg(not(windows))]
const LIB_NAME: &str = "libOpenCL.so.1";

cl_api! {
    clGetPlatformIDs: unsafe extern "system" fn(u32, *mut H, *mut u32) -> i32,
    clGetPlatformInfo: unsafe extern "system" fn(H, u32, usize, *mut c_void, *mut usize) -> i32,
    clGetDeviceIDs: unsafe extern "system" fn(H, u64, u32, *mut H, *mut u32) -> i32,
    clGetDeviceInfo: unsafe extern "system" fn(H, u32, usize, *mut c_void, *mut usize) -> i32,
    clCreateContext: unsafe extern "system" fn(*const isize, u32, *const H, *const c_void, *mut c_void, *mut i32) -> H,
    clCreateCommandQueue: unsafe extern "system" fn(H, H, u64, *mut i32) -> H,
    clCreateProgramWithSource: unsafe extern "system" fn(H, u32, *const *const c_char, *const usize, *mut i32) -> H,
    clBuildProgram: unsafe extern "system" fn(H, u32, *const H, *const c_char, *const c_void, *mut c_void) -> i32,
    clGetProgramBuildInfo: unsafe extern "system" fn(H, H, u32, usize, *mut c_void, *mut usize) -> i32,
    clCreateKernel: unsafe extern "system" fn(H, *const c_char, *mut i32) -> H,
    clSetKernelArg: unsafe extern "system" fn(H, u32, usize, *const c_void) -> i32,
    clEnqueueNDRangeKernel: unsafe extern "system" fn(H, H, u32, *const usize, *const usize, *const usize, u32, *const H, *mut H) -> i32,
    clCreateBuffer: unsafe extern "system" fn(H, u64, usize, *mut c_void, *mut i32) -> H,
    clReleaseMemObject: unsafe extern "system" fn(H) -> i32,
    clEnqueueReadBuffer: unsafe extern "system" fn(H, H, u32, usize, usize, *mut c_void, u32, *const H, *mut H) -> i32,
    clEnqueueWriteBuffer: unsafe extern "system" fn(H, H, u32, usize, usize, *const c_void, u32, *const H, *mut H) -> i32,
    clEnqueueCopyBuffer: unsafe extern "system" fn(H, H, H, usize, usize, usize, u32, *const H, *mut H) -> i32,
    clFlush: unsafe extern "system" fn(H) -> i32,
    clFinish: unsafe extern "system" fn(H) -> i32,
}

// ---- devices ------------------------------------------------------------------------

struct State {
    kernels: Option<HashMap<&'static str, H>>,
    scratch: H,
}

/// Reusable buffers. Handles are stored as `usize` so the cache is `Send`.
#[derive(Default)]
struct Cache {
    free: HashMap<usize, Vec<usize>>,
    /// Size of every live-or-cached buffer we created, by handle.
    sizes: HashMap<usize, usize>,
    cached_bytes: u64,
}

struct Dev {
    ctx: H,
    queue: H,
    dev: H,
    name: String,
    total: u64,
    max_alloc: u64,
    /// Integrated GPU: allocations come out of system RAM (`CL_DEVICE_HOST_UNIFIED_MEMORY`).
    shared: bool,
    /// Bytes of all buffers created and not yet released to the runtime (in use + cached).
    allocated: AtomicU64,
    cache: Mutex<Cache>,
    host_ptr: bool,
    /// Serializes queue / kernel-argument access.
    state: Mutex<State>,
}

struct Global {
    cl: Cl,
    devs: Vec<Dev>,
}

// Raw OpenCL handles are thread-safe objects; kernel args/queue use is serialized by `Dev::state`.
unsafe impl Send for Global {}
unsafe impl Sync for Global {}

static GLOBAL: OnceLock<Option<Global>> = OnceLock::new();

fn global() -> Option<&'static Global> {
    GLOBAL.get_or_init(|| catch_unwind(|| unsafe { init() }).ok().flatten()).as_ref()
}

fn verbose() -> bool {
    std::env::var_os("PYTORCHES_XPU_VERBOSE").is_some()
}

unsafe fn info_string(f: unsafe extern "system" fn(H, u32, usize, *mut c_void, *mut usize) -> i32, h: H, param: u32) -> String {
    unsafe {
        let mut buf = [0u8; 512];
        let mut len = 0usize;
        if f(h, param, buf.len(), buf.as_mut_ptr() as *mut c_void, &mut len) != 0 {
            return String::new();
        }
        String::from_utf8_lossy(&buf[..len.saturating_sub(1).min(buf.len())]).into_owned()
    }
}

unsafe fn dev_u64(cl: &Cl, d: H, param: u32) -> u64 {
    let mut v = 0u64;
    unsafe { (cl.clGetDeviceInfo)(d, param, 8, &mut v as *mut u64 as *mut c_void, null_mut()) };
    v
}

unsafe fn init() -> Option<Global> {
    unsafe {
        let cl = Cl::load().ok()?;
        let mut np = 0u32;
        if (cl.clGetPlatformIDs)(0, null_mut(), &mut np) != 0 || np == 0 {
            return None;
        }
        let mut plats = vec![null_mut(); np as usize];
        if (cl.clGetPlatformIDs)(np, plats.as_mut_ptr(), null_mut()) != 0 {
            return None;
        }
        // Platforms whose name mentions Intel first.
        let mut named: Vec<(bool, H, String)> = plats
            .iter()
            .map(|&p| {
                let n = info_string(cl.clGetPlatformInfo, p, CL_PLATFORM_NAME);
                (!n.contains("Intel"), p, n)
            })
            .collect();
        named.sort_by_key(|t| t.0);

        let host_ptr = std::env::var("PYTORCHES_XPU_ALLOC").map(|v| v == "host").unwrap_or(false);
        for (_, plat, pname) in named {
            let mut nd = 0u32;
            if (cl.clGetDeviceIDs)(plat, CL_DEVICE_TYPE_GPU, 0, null_mut(), &mut nd) != 0 || nd == 0 {
                continue;
            }
            let mut ids = vec![null_mut(); nd as usize];
            if (cl.clGetDeviceIDs)(plat, CL_DEVICE_TYPE_GPU, nd, ids.as_mut_ptr(), null_mut()) != 0 {
                continue;
            }
            let mut devs = Vec::new();
            for d in ids {
                if dev_u64(&cl, d, CL_DEVICE_VENDOR_ID) & 0xFFFF_FFFF != INTEL_VENDOR_ID {
                    continue;
                }
                let mut e = 0i32;
                let ctx = (cl.clCreateContext)(null(), 1, &d, null(), null_mut(), &mut e);
                if ctx.is_null() || e != 0 {
                    continue;
                }
                let queue = (cl.clCreateCommandQueue)(ctx, d, 0, &mut e);
                if queue.is_null() || e != 0 {
                    continue;
                }
                let name = info_string(cl.clGetDeviceInfo, d, CL_DEVICE_NAME);
                let total = dev_u64(&cl, d, CL_DEVICE_GLOBAL_MEM_SIZE);
                let max_alloc = dev_u64(&cl, d, CL_DEVICE_MAX_MEM_ALLOC_SIZE);
                let shared = dev_u64(&cl, d, CL_DEVICE_HOST_UNIFIED_MEMORY) & 0xFFFF_FFFF != 0;
                if verbose() {
                    eprintln!(
                        "[xpu] platform '{pname}' device '{name}': global_mem={} MiB max_alloc={} MiB host_ptr={host_ptr}",
                        total >> 20,
                        max_alloc >> 20
                    );
                }
                devs.push(Dev {
                    ctx,
                    queue,
                    dev: d,
                    name,
                    total,
                    max_alloc,
                    shared,
                    allocated: AtomicU64::new(0),
                    cache: Mutex::new(Cache::default()),
                    host_ptr,
                    state: Mutex::new(State { kernels: None, scratch: null_mut() }),
                });
            }
            if !devs.is_empty() {
                return Some(Global { cl, devs });
            }
        }
        None
    }
}

fn dev_of(device: u32) -> R<(&'static Global, &'static Dev)> {
    let Some(g) = global() else { return fail(STATUS_INVALID_ARGUMENT, "no Intel GPU available") };
    match g.devs.get(device as usize) {
        Some(d) => Ok((g, d)),
        None => fail(STATUS_INVALID_ARGUMENT, format!("bad device index {device}")),
    }
}

fn lock(d: &Dev) -> std::sync::MutexGuard<'_, State> {
    d.state.lock().unwrap_or_else(|p| p.into_inner())
}

// ---- program / kernels ----------------------------------------------------------------

/// Kernels that exist only when the driver supports an extension they need; missing ones are skipped.
const OPTIONAL_KERNELS: &[&str] = &["matmul_sg"];

const KERNEL_NAMES: &[&str] = &[
    "unary_flat",
    "unary_strided",
    "binary_flat",
    "binary_strided",
    "axpy",
    "transpose2d",
    "fill",
    "rand_normal",
    "matmul",
    "matmul_sg",
    "sum_rows",
    "sum_inner",
    "reduce_partials",
];

unsafe fn ensure_ready(g: &Global, d: &Dev, st: &mut State) -> R<()> {
    unsafe {
        if st.kernels.is_some() {
            return Ok(());
        }
        let mut e = 0i32;
        let src = CString::new(KERNEL_SRC).unwrap();
        let ptr = src.as_ptr();
        let len = KERNEL_SRC.len();
        let prog = (g.cl.clCreateProgramWithSource)(d.ctx, 1, &ptr, &len, &mut e);
        check(e, "clCreateProgramWithSource")?;
        let opts = std::env::var("PYTORCHES_XPU_BUILD_OPTS").unwrap_or_else(|_| DEFAULT_BUILD_OPTS.to_string());
        let copts = CString::new(opts.clone()).unwrap();
        let rc = (g.cl.clBuildProgram)(prog, 1, &d.dev, copts.as_ptr(), null(), null_mut());
        if rc != 0 {
            let mut size = 0usize;
            (g.cl.clGetProgramBuildInfo)(prog, d.dev, CL_PROGRAM_BUILD_LOG, 0, null_mut(), &mut size);
            let mut log = vec![0u8; size.max(1)];
            (g.cl.clGetProgramBuildInfo)(prog, d.dev, CL_PROGRAM_BUILD_LOG, size, log.as_mut_ptr() as *mut c_void, null_mut());
            let log = String::from_utf8_lossy(&log).trim_end_matches('\0').to_string();
            return fail(
                STATUS_INTERNAL,
                format!("clBuildProgram (options '{opts}') failed: {} ({rc})\n{log}", cl_err_name(rc)),
            );
        }
        let mut kernels = HashMap::new();
        for &name in KERNEL_NAMES {
            let cname = CString::new(name).unwrap();
            let k = (g.cl.clCreateKernel)(prog, cname.as_ptr(), &mut e);
            if e != 0 && OPTIONAL_KERNELS.contains(&name) {
                continue;
            }
            check(e, &format!("clCreateKernel({name})"))?;
            kernels.insert(name, k);
        }
        let scratch = (g.cl.clCreateBuffer)(d.ctx, CL_MEM_READ_WRITE, SCRATCH_FLOATS * 4, null_mut(), &mut e);
        check(e, "scratch buffer")?;
        st.scratch = scratch;
        st.kernels = Some(kernels);
        Ok(())
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DimsC {
    ndim: u32,
    pad: u32,
    shape: [u64; 8],
    stride: [u64; 8],
}

#[derive(Clone, Copy)]
enum Arg {
    Mem(H),
    U64(u64),
    I32(i32),
    F32(f32),
    Dims(DimsC),
}

unsafe fn launch(
    g: &Global,
    d: &Dev,
    st: &State,
    name: &str,
    args: &[Arg],
    global_size: &[usize],
    local_size: Option<&[usize]>,
) -> R<()> {
    unsafe {
        let k = *st.kernels.as_ref().unwrap().get(name).expect("kernel");
        for (i, a) in args.iter().enumerate() {
            let rc = match a {
                Arg::Mem(h) => (g.cl.clSetKernelArg)(k, i as u32, size_of::<H>(), h as *const H as *const c_void),
                Arg::U64(v) => (g.cl.clSetKernelArg)(k, i as u32, 8, v as *const u64 as *const c_void),
                Arg::I32(v) => (g.cl.clSetKernelArg)(k, i as u32, 4, v as *const i32 as *const c_void),
                Arg::F32(v) => (g.cl.clSetKernelArg)(k, i as u32, 4, v as *const f32 as *const c_void),
                Arg::Dims(v) => (g.cl.clSetKernelArg)(k, i as u32, size_of::<DimsC>(), v as *const DimsC as *const c_void),
            };
            check(rc, &format!("clSetKernelArg({name}, {i})"))?;
        }
        let local = local_size.map(|l| l.as_ptr()).unwrap_or(null());
        let rc = (g.cl.clEnqueueNDRangeKernel)(
            d.queue,
            k,
            global_size.len() as u32,
            null(),
            global_size.as_ptr(),
            local,
            0,
            null(),
            null_mut(),
        );
        check(rc, &format!("clEnqueueNDRangeKernel({name})"))?;
        // Make sure the batch actually starts running; do not wait for it.
        (g.cl.clFlush)(d.queue);
        Ok(())
    }
}

// ---- memory entry points --------------------------------------------------------------

/// Rounds a request up so similar sizes share buffers: powers of two below 1 MiB, 2 MiB steps above.
fn bucket(bytes: usize) -> usize {
    const MIB: usize = 1 << 20;
    if bytes < MIB { bytes.max(256).next_power_of_two() } else { bytes.div_ceil(2 * MIB) * 2 * MIB }
}

fn cache_lock(d: &Dev) -> std::sync::MutexGuard<'_, Cache> {
    d.cache.lock().unwrap_or_else(|p| p.into_inner())
}

/// Creates a buffer of exactly `bytes`, or returns the OpenCL error code.
unsafe fn create_buffer(g: &Global, d: &Dev, bytes: usize) -> Result<H, i32> {
    unsafe {
        let mut base = CL_MEM_READ_WRITE;
        if d.host_ptr {
            base |= CL_MEM_ALLOC_HOST_PTR;
        }
        let big = bytes as u64 > d.max_alloc || bytes as u64 >= FOUR_GIB;
        let flags = if big { base | CL_MEM_ALLOW_UNRESTRICTED_SIZE_INTEL } else { base };
        let mut e = 0i32;
        let mut mem = (g.cl.clCreateBuffer)(d.ctx, flags, bytes, null_mut(), &mut e);
        if (mem.is_null() || e != 0) && !big {
            // Some drivers report a smaller limit than they really allow; retry unrestricted.
            e = 0;
            mem = (g.cl.clCreateBuffer)(d.ctx, base | CL_MEM_ALLOW_UNRESTRICTED_SIZE_INTEL, bytes, null_mut(), &mut e);
        }
        if mem.is_null() || e != 0 { Err(if e == 0 { -4 } else { e }) } else { Ok(mem) }
    }
}

/// Returns every cached buffer to the runtime. The runtime defers destruction until queued work using
/// a buffer has finished, so this is safe with kernels still in flight.
unsafe fn flush_cache(g: &Global, d: &Dev, c: &mut Cache) {
    unsafe {
        for (size, handles) in c.free.drain() {
            for h in handles {
                (g.cl.clReleaseMemObject)(h as H);
                c.sizes.remove(&h);
                c.cached_bytes -= size as u64;
                d.allocated.fetch_sub(size as u64, Ordering::Relaxed);
            }
        }
    }
}

unsafe extern "C" fn alloc_buf(device: u32, bytes: usize, out: *mut *mut c_void) -> Status {
    guard(|| unsafe {
        let (g, d) = dev_of(device)?;
        if out.is_null() {
            return fail(STATUS_INVALID_ARGUMENT, "null out pointer");
        }
        let bytes = bytes.max(4);
        let size = bucket(bytes);
        let mut c = cache_lock(d);
        if let Some(h) = c.free.get_mut(&size).and_then(|v| v.pop()) {
            c.cached_bytes -= size as u64;
            *out = h as H;
            return Ok(());
        }
        let mut got = size;
        let mut res = create_buffer(g, d, size);
        if res.is_err() {
            // Out of memory (or a refused size): give the cache back and try again, then try the
            // exact size without the bucket rounding.
            flush_cache(g, d, &mut c);
            res = create_buffer(g, d, size);
            if res.is_err() && bytes < size {
                got = bytes;
                res = create_buffer(g, d, bytes);
            }
        }
        match res {
            Ok(mem) => {
                c.sizes.insert(mem as usize, got);
                d.allocated.fetch_add(got as u64, Ordering::Relaxed);
                *out = mem;
                Ok(())
            }
            Err(e) => {
                let status = if is_oom_code(e) || e == -61 { STATUS_OUT_OF_MEMORY } else { STATUS_INTERNAL };
                fail(
                    status,
                    format!(
                        "clCreateBuffer({bytes} bytes) failed: {} ({e}); tracked allocations {} bytes, device total {} bytes, max single alloc {} bytes",
                        cl_err_name(e),
                        d.allocated.load(Ordering::Relaxed),
                        d.total,
                        d.max_alloc
                    ),
                )
            }
        }
    })
}

unsafe extern "C" fn free_buf(device: u32, ptr: *mut c_void) {
    if ptr.is_null() {
        return;
    }
    let _ = guard(|| {
        let (_, d) = dev_of(device)?;
        let mut c = cache_lock(d);
        // Buffers are reused, not released: in-order queue semantics make reuse safe while earlier work
        // that touched the buffer is still in flight.
        if let Some(&size) = c.sizes.get(&(ptr as usize)) {
            c.free.entry(size).or_default().push(ptr as usize);
            c.cached_bytes += size as u64;
        }
        Ok(())
    });
}

unsafe extern "C" fn copy_from_host(device: u32, dst: *mut c_void, src: *const c_void, bytes: usize) -> Status {
    guard(|| unsafe {
        let (g, d) = dev_of(device)?;
        let _st = lock(d);
        let mut off = 0usize;
        while off < bytes {
            let n = (bytes - off).min(XFER_CHUNK);
            let rc = (g.cl.clEnqueueWriteBuffer)(
                d.queue,
                dst,
                1,
                off,
                n,
                (src as *const u8).add(off) as *const c_void,
                0,
                null(),
                null_mut(),
            );
            check(rc, "clEnqueueWriteBuffer")?;
            off += n;
        }
        Ok(())
    })
}

unsafe extern "C" fn copy_to_host(device: u32, dst: *mut c_void, src: *const c_void, bytes: usize) -> Status {
    guard(|| unsafe {
        let (g, d) = dev_of(device)?;
        let _st = lock(d);
        let mut off = 0usize;
        while off < bytes {
            let n = (bytes - off).min(XFER_CHUNK);
            let rc = (g.cl.clEnqueueReadBuffer)(
                d.queue,
                src as H,
                1,
                off,
                n,
                (dst as *mut u8).add(off) as *mut c_void,
                0,
                null(),
                null_mut(),
            );
            check(rc, "clEnqueueReadBuffer")?;
            off += n;
        }
        Ok(())
    })
}

unsafe extern "C" fn copy_d2d(device: u32, dst: *mut c_void, src: *const c_void, bytes: usize) -> Status {
    guard(|| unsafe {
        let (g, d) = dev_of(device)?;
        let _st = lock(d);
        let mut off = 0usize;
        while off < bytes {
            let n = (bytes - off).min(XFER_CHUNK);
            let rc = (g.cl.clEnqueueCopyBuffer)(d.queue, src as H, dst, off, off, n, 0, null(), null_mut());
            check(rc, "clEnqueueCopyBuffer")?;
            off += n;
        }
        (g.cl.clFlush)(d.queue);
        Ok(())
    })
}

// ---- device info / sync ----------------------------------------------------------------

unsafe extern "C" fn device_count() -> u32 {
    catch_unwind(|| global().map(|g| g.devs.len() as u32).unwrap_or(0)).unwrap_or(0)
}

unsafe extern "C" fn device_info(device: u32, out: *mut DeviceInfo) -> Status {
    guard(|| unsafe {
        let (_, d) = dev_of(device)?;
        if out.is_null() {
            return fail(STATUS_INVALID_ARGUMENT, "null out pointer");
        }
        let mut info = DeviceInfo { name: [0; 64], kind: KIND_XPU, total_memory: d.total, free_memory: 0, flags: 0 };
        if d.shared {
            info.flags |= DEVICE_FLAG_SHARED_HOST_MEMORY;
        }
        for (dst, &b) in info.name.iter_mut().zip(d.name.as_bytes().iter().take(63)) {
            *dst = b as c_char;
        }
        // OpenCL has no free-memory query: this is total minus what this plugin holds that is not
        // reusable (cached buffers are free to the next allocation).
        let cached = cache_lock(d).cached_bytes;
        let in_use = d.allocated.load(Ordering::Relaxed).saturating_sub(cached);
        info.free_memory = d.total.saturating_sub(in_use);
        *out = info;
        Ok(())
    })
}

unsafe extern "C" fn synchronize(device: u32) -> Status {
    guard(|| unsafe {
        let (g, d) = dev_of(device)?;
        let _st = lock(d);
        check((g.cl.clFinish)(d.queue), "clFinish")
    })
}

unsafe extern "C" fn last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

// ---- ops ----------------------------------------------------------------------------------

unsafe extern "C" fn supports_op(code: u32) -> u32 {
    use op::*;
    matches!(
        code,
        NEG | EXP | LOG | RELU | TANH | STEP | ADD | SUB | MUL | DIV | MATMUL | AXPY | SUM_AXIS | COPY | FILL | RAND_NORMAL
    ) as u32
}

struct T {
    data: H,
    shape: Vec<u64>,
    strides: Vec<u64>,
}

impl T {
    fn numel(&self) -> u64 {
        self.shape.iter().product()
    }
    fn is_contiguous(&self) -> bool {
        let mut expect = 1u64;
        for i in (0..self.shape.len()).rev() {
            if self.shape[i] != 1 && self.strides[i] != expect {
                return false;
            }
            expect *= self.shape[i];
        }
        true
    }
}

unsafe fn tdesc(d: &TensorDesc) -> R<T> {
    unsafe {
        if d.dtype != DTYPE_F32 {
            return fail(STATUS_UNSUPPORTED, format!("unsupported dtype {}", d.dtype));
        }
        if d.data.is_null() || (d.ndim > 0 && (d.shape.is_null() || d.strides.is_null())) {
            return fail(STATUS_INVALID_ARGUMENT, "null tensor data/shape/strides");
        }
        Ok(T {
            data: d.data,
            shape: slice::from_raw_parts(d.shape, d.ndim as usize).to_vec(),
            strides: slice::from_raw_parts(d.strides, d.ndim as usize).to_vec(),
        })
    }
}

fn ew_global(n: u64) -> usize {
    (n.div_ceil(256).min(1 << 18) * 256) as usize
}

/// Result of coalescing the dims of an elementwise op.
struct Plan {
    flat: bool,
    use32: bool,
    dims: Vec<DimsC>,
}

/// Drops size-1 dims and merges dims that are jointly contiguous for every operand.
fn plan(shape: &[u64], strides: &[&[u64]]) -> R<Plan> {
    let mut merged: Vec<(u64, Vec<u64>)> = Vec::new();
    for (k, &s) in shape.iter().enumerate() {
        if s == 1 {
            continue;
        }
        let st: Vec<u64> = strides.iter().map(|x| x[k]).collect();
        if let Some(last) = merged.last_mut() {
            if (0..st.len()).all(|j| last.1[j] == st[j].wrapping_mul(s)) {
                last.0 *= s;
                last.1 = st;
                continue;
            }
        }
        merged.push((s, st));
    }
    if merged.len() > 8 {
        return fail(STATUS_UNSUPPORTED, "more than 8 non-coalescable dimensions");
    }
    let n: u64 = shape.iter().product();
    let flat = merged.is_empty() || (merged.len() == 1 && merged[0].1.iter().all(|&s| s == 1));
    let mut use32 = n < (1 << 32);
    for j in 0..strides.len() {
        let reach: u128 = merged.iter().map(|(s, st)| (*s as u128 - 1) * st[j] as u128).sum();
        if reach >= (1u128 << 32) {
            use32 = false;
        }
    }
    let dims = (0..strides.len())
        .map(|j| {
            let mut d = DimsC { ndim: merged.len() as u32, pad: 0, shape: [1; 8], stride: [0; 8] };
            for (k, (s, st)) in merged.iter().enumerate() {
                d.shape[k] = *s;
                d.stride[k] = st[j];
            }
            d
        })
        .collect();
    Ok(Plan { flat, use32, dims })
}

unsafe fn run(device: u32, op_code: u32, attrs: &OpAttrs, ins: &[T], outs: &[T]) -> R<()> {
    use op::*;
    unsafe {
        let (g, d) = dev_of(device)?;
        if outs.len() != 1 {
            return fail(STATUS_INVALID_ARGUMENT, "expected exactly one output");
        }
        let out = &outs[0];
        let n = out.numel();
        let need_in = match op_code {
            FILL | RAND_NORMAL => 0,
            NEG | EXP | LOG | RELU | TANH | STEP | COPY | SUM_AXIS => 1,
            ADD | SUB | MUL | DIV | MATMUL | AXPY => 2,
            _ => return fail(STATUS_UNSUPPORTED, format!("unsupported op {op_code}")),
        };
        if ins.len() != need_in {
            return fail(STATUS_INVALID_ARGUMENT, format!("op {op_code} expects {need_in} inputs, got {}", ins.len()));
        }
        for t in ins {
            if t.shape.len() != t.strides.len() {
                return fail(STATUS_INVALID_ARGUMENT, "shape/strides mismatch");
            }
        }
        if n == 0 && op_code != SUM_AXIS && op_code != MATMUL {
            return Ok(());
        }
        let mut st = lock(d);
        ensure_ready(g, d, &mut st)?;
        let st = &*st;

        match op_code {
            FILL => {
                let v = f32::from_bits(attrs.ints[0] as u32);
                launch(g, d, st, "fill", &[Arg::Mem(out.data), Arg::U64(n), Arg::F32(v)], &[ew_global(n)], Some(&[256]))
            }
            RAND_NORMAL => {
                let seed = attrs.ints[0] as u64;
                launch(
                    g,
                    d,
                    st,
                    "rand_normal",
                    &[Arg::Mem(out.data), Arg::U64(n), Arg::U64(seed)],
                    &[ew_global(n)],
                    Some(&[256]),
                )
            }
            NEG | EXP | LOG | RELU | TANH | STEP | COPY => {
                let x = &ins[0];
                if x.shape != out.shape {
                    return fail(STATUS_INVALID_ARGUMENT, "unary input/output shape mismatch");
                }
                // A COPY that is exactly a 2-D transpose of a contiguous matrix (what `Tensor::t` and the
                // matmul backward produce) takes the tiled kernel instead of the generic strided one.
                if op_code == COPY && x.shape.len() == 2 && x.strides[0] == 1 && x.strides[1] == x.shape[0] && x.shape[0] > 1 && x.shape[1] > 1 {
                    let (c, r) = (x.shape[0], x.shape[1]); // source is [r, c]; output is [c, r]
                    return launch(
                        g,
                        d,
                        st,
                        "transpose2d",
                        &[Arg::Mem(x.data), Arg::Mem(out.data), Arg::U64(r), Arg::U64(c)],
                        &[c.div_ceil(32) as usize * 32, r.div_ceil(32) as usize * 8],
                        Some(&[32, 8]),
                    );
                }
                let code = if op_code == COPY { 0 } else { op_code as i32 };
                let p = plan(&x.shape, &[&x.strides])?;
                if p.flat {
                    launch(
                        g,
                        d,
                        st,
                        "unary_flat",
                        &[Arg::Mem(x.data), Arg::Mem(out.data), Arg::U64(n), Arg::I32(code)],
                        &[ew_global(n)],
                        Some(&[256]),
                    )
                } else {
                    launch(
                        g,
                        d,
                        st,
                        "unary_strided",
                        &[
                            Arg::Mem(x.data),
                            Arg::Mem(out.data),
                            Arg::U64(n),
                            Arg::I32(code),
                            Arg::Dims(p.dims[0]),
                            Arg::I32(p.use32 as i32),
                        ],
                        &[ew_global(n)],
                        Some(&[256]),
                    )
                }
            }
            ADD | SUB | MUL | DIV => {
                let (a, b) = (&ins[0], &ins[1]);
                if a.shape != out.shape || b.shape != out.shape {
                    return fail(STATUS_INVALID_ARGUMENT, "binary input/output shape mismatch");
                }
                let code = (op_code - ADD) as i32;
                let p = plan(&out.shape, &[&a.strides, &b.strides])?;
                if p.flat {
                    launch(
                        g,
                        d,
                        st,
                        "binary_flat",
                        &[Arg::Mem(a.data), Arg::Mem(b.data), Arg::Mem(out.data), Arg::U64(n), Arg::I32(code)],
                        &[ew_global(n)],
                        Some(&[256]),
                    )
                } else {
                    launch(
                        g,
                        d,
                        st,
                        "binary_strided",
                        &[
                            Arg::Mem(a.data),
                            Arg::Mem(b.data),
                            Arg::Mem(out.data),
                            Arg::U64(n),
                            Arg::I32(code),
                            Arg::Dims(p.dims[0]),
                            Arg::Dims(p.dims[1]),
                            Arg::I32(p.use32 as i32),
                        ],
                        &[ew_global(n)],
                        Some(&[256]),
                    )
                }
            }
            AXPY => {
                let (a, b) = (&ins[0], &ins[1]);
                if a.numel() != n || b.numel() != n || !a.is_contiguous() || !b.is_contiguous() {
                    return fail(STATUS_INVALID_ARGUMENT, "axpy needs contiguous inputs matching the output");
                }
                let alpha = f32::from_bits(attrs.ints[0] as u32);
                launch(
                    g,
                    d,
                    st,
                    "axpy",
                    &[Arg::Mem(a.data), Arg::Mem(b.data), Arg::Mem(out.data), Arg::U64(n), Arg::F32(alpha)],
                    &[ew_global(n)],
                    Some(&[256]),
                )
            }
            MATMUL => {
                let (a, b) = (&ins[0], &ins[1]);
                if a.shape.len() != 2 || b.shape.len() != 2 || out.shape.len() != 2 {
                    return fail(STATUS_INVALID_ARGUMENT, "matmul needs 2-D operands");
                }
                let (m, k, nn) = (a.shape[0], a.shape[1], b.shape[1]);
                if b.shape[0] != k || out.shape[0] != m || out.shape[1] != nn {
                    return fail(STATUS_INVALID_ARGUMENT, format!("matmul shape mismatch {:?} x {:?} -> {:?}", a.shape, b.shape, out.shape));
                }
                if !a.is_contiguous() || !b.is_contiguous() {
                    return fail(STATUS_INVALID_ARGUMENT, "matmul inputs must be contiguous");
                }
                if m == 0 || nn == 0 {
                    return Ok(());
                }
                // Sub-group fast path (see kernels.cl) for N % 32 == 0 and K % 16 == 0 on whole 16-row
                // blocks, when the driver has it. Leftover rows (m % 16) use the general kernel.
                let mut done = 0u64;
                if nn % 32 == 0 && k % 16 == 0 && k > 0 && m >= 16 && st.kernels.as_ref().unwrap().contains_key("matmul_sg") {
                    let fast = m / 16 * 16;
                    launch(
                        g,
                        d,
                        st,
                        "matmul_sg",
                        &[Arg::Mem(a.data), Arg::Mem(b.data), Arg::Mem(out.data), Arg::U64(fast), Arg::U64(nn), Arg::U64(k)],
                        &[(nn / 32) as usize * 16, (fast / 16) as usize],
                        Some(&[16, 1]),
                    )?;
                    done = fast;
                }
                let rest = m - done;
                if rest == 0 {
                    return Ok(());
                }
                let gx = nn.div_ceil(64) as usize * 16;
                let gy = rest.div_ceil(64) as usize * 16;
                launch(
                    g,
                    d,
                    st,
                    "matmul",
                    &[Arg::Mem(a.data), Arg::Mem(b.data), Arg::Mem(out.data), Arg::U64(rest), Arg::U64(nn), Arg::U64(k), Arg::U64(done)],
                    &[gx, gy],
                    Some(&[16, 16]),
                )
            }
            SUM_AXIS => {
                let x = &ins[0];
                let axis = attrs.ints[0];
                if axis < 0 || axis as usize >= x.shape.len() {
                    return fail(STATUS_INVALID_ARGUMENT, "sum axis out of range");
                }
                let axis = axis as usize;
                if !x.is_contiguous() {
                    return fail(STATUS_INVALID_ARGUMENT, "sum input must be contiguous");
                }
                let outer: u64 = x.shape[..axis].iter().product();
                let len = x.shape[axis];
                let inner: u64 = x.shape[axis + 1..].iter().product();
                let outs_n = outer * inner;
                if out.numel() != outs_n {
                    return fail(STATUS_INVALID_ARGUMENT, "sum output size mismatch");
                }
                if outs_n == 0 {
                    return Ok(());
                }
                if inner == 1 && len >= 512 {
                    // Work-group tree reduction, optionally split across groups with a second pass.
                    let splits = if outer >= 512 { 1 } else { 512u64.div_ceil(outer).min((len / 8192).max(1)).min(512) };
                    let chunk = len.div_ceil(splits);
                    let groups = (outer * splits).min(1 << 20) as usize;
                    let dst = if splits == 1 { out.data } else { st.scratch };
                    launch(
                        g,
                        d,
                        st,
                        "sum_rows",
                        &[Arg::Mem(x.data), Arg::Mem(dst), Arg::U64(outer), Arg::U64(len), Arg::U64(splits), Arg::U64(chunk)],
                        &[groups * 256],
                        Some(&[256]),
                    )?;
                    if splits > 1 {
                        launch(
                            g,
                            d,
                            st,
                            "reduce_partials",
                            &[Arg::Mem(st.scratch), Arg::Mem(out.data), Arg::U64(outs_n), Arg::U64(splits)],
                            &[ew_global(outs_n)],
                            Some(&[256]),
                        )?;
                    }
                    Ok(())
                } else {
                    let splits = if outs_n >= 32768 { 1 } else { 32768u64.div_ceil(outs_n).min((len / 64).max(1)).min(4096) };
                    let chunk = len.div_ceil(splits).max(1);
                    if (outs_n * splits) as usize > SCRATCH_FLOATS && splits > 1 {
                        return fail(STATUS_INTERNAL, "sum scratch overflow");
                    }
                    let total = outs_n * splits;
                    let dst = if splits == 1 { out.data } else { st.scratch };
                    launch(
                        g,
                        d,
                        st,
                        "sum_inner",
                        &[
                            Arg::Mem(x.data),
                            Arg::Mem(dst),
                            Arg::U64(outer),
                            Arg::U64(len),
                            Arg::U64(inner),
                            Arg::U64(splits),
                            Arg::U64(chunk),
                        ],
                        &[ew_global(total)],
                        Some(&[256]),
                    )?;
                    if splits > 1 {
                        launch(
                            g,
                            d,
                            st,
                            "reduce_partials",
                            &[Arg::Mem(st.scratch), Arg::Mem(out.data), Arg::U64(outs_n), Arg::U64(splits)],
                            &[ew_global(outs_n)],
                            Some(&[256]),
                        )?;
                    }
                    Ok(())
                }
            }
            _ => fail(STATUS_UNSUPPORTED, format!("unsupported op {op_code}")),
        }
    }
}

unsafe extern "C" fn execute(
    device: u32,
    op_code: u32,
    attrs: *const OpAttrs,
    inputs: *const TensorDesc,
    n_inputs: u32,
    outputs: *const TensorDesc,
    n_outputs: u32,
) -> Status {
    guard(|| unsafe {
        let attrs = if attrs.is_null() { OpAttrs::default() } else { *attrs };
        let ins: Vec<T> = if n_inputs == 0 {
            Vec::new()
        } else {
            slice::from_raw_parts(inputs, n_inputs as usize).iter().map(|d| tdesc(d)).collect::<R<_>>()?
        };
        let outs: Vec<T> = if n_outputs == 0 {
            Vec::new()
        } else {
            slice::from_raw_parts(outputs, n_outputs as usize).iter().map(|d| tdesc(d)).collect::<R<_>>()?
        };
        run(device, op_code, &attrs, &ins, &outs)
    })
}

// ---- entry point -----------------------------------------------------------------------

static VTABLE: PluginVTable = PluginVTable {
    abi_version: ABI_VERSION,
    plugin_version: 1,
    name: c"xpu".as_ptr(),
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

#[allow(dead_code)]
fn _unused(_: &CStr) {}
