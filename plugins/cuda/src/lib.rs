//! NVIDIA CUDA plugin. Uses the CUDA *driver* API loaded at runtime from nvcuda.dll (libcuda.so.1),
//! so only the NVIDIA driver is needed. Kernels are embedded PTX, JIT-compiled by the driver.
//! Everything runs on the primary context's default (NULL) stream.
//!
//! Matmul uses cuBLAS when it can be found at runtime (also loaded dynamically, so nothing is linked
//! at build time) and falls back to our own tiled kernel otherwise. `PYTORCHES_CUBLAS` overrides the
//! lookup: `0` disables cuBLAS, anything else is the path of the library to load.

use libloading::Library;
use pytorches_plugin_abi::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::slice;
use std::sync::{Arc, Mutex, OnceLock};

static PTX: &str = include_str!("../kernels/kernels.ptx");

// ---- errors -------------------------------------------------------------------------

struct E {
    status: Status,
    msg: String,
}

fn err<T>(status: Status, msg: impl Into<String>) -> Result<T, E> {
    Result::Err(E { status, msg: msg.into() })
}

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_last_error(msg: &str) {
    let c = CString::new(msg.replace('\0', " ")).unwrap_or_default();
    LAST_ERROR.with(|e| *e.borrow_mut() = c);
}

/// Runs an entry point body: converts errors and panics to a Status plus a thread-local message.
fn wrap(f: impl FnOnce() -> Result<(), E>) -> Status {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => STATUS_OK,
        Ok(Result::Err(e)) => {
            set_last_error(&format!("cuda plugin: {}", e.msg));
            e.status
        }
        Err(_) => {
            set_last_error("cuda plugin: internal panic");
            STATUS_INTERNAL
        }
    }
}

// ---- driver API ----------------------------------------------------------------------

type CuResult = c_int;
type CuDevice = c_int;
type CuPtr = u64;
type Handle = *mut c_void;
const CUDA_ERROR_OUT_OF_MEMORY: CuResult = 2;

struct Driver {
    _lib: Library,
    cu_init: unsafe extern "C" fn(c_uint) -> CuResult,
    device_get_count: unsafe extern "C" fn(*mut c_int) -> CuResult,
    device_get: unsafe extern "C" fn(*mut CuDevice, c_int) -> CuResult,
    device_get_name: unsafe extern "C" fn(*mut c_char, c_int, CuDevice) -> CuResult,
    primary_retain: unsafe extern "C" fn(*mut Handle, CuDevice) -> CuResult,
    ctx_set_current: unsafe extern "C" fn(Handle) -> CuResult,
    ctx_synchronize: unsafe extern "C" fn() -> CuResult,
    mem_alloc: unsafe extern "C" fn(*mut CuPtr, usize) -> CuResult,
    mem_free: unsafe extern "C" fn(CuPtr) -> CuResult,
    htod: unsafe extern "C" fn(CuPtr, *const c_void, usize) -> CuResult,
    dtoh: unsafe extern "C" fn(*mut c_void, CuPtr, usize) -> CuResult,
    dtod: unsafe extern "C" fn(CuPtr, CuPtr, usize) -> CuResult,
    mem_get_info: unsafe extern "C" fn(*mut usize, *mut usize) -> CuResult,
    module_load_ex: unsafe extern "C" fn(*mut Handle, *const c_void, c_uint, *mut c_int, *mut *mut c_void) -> CuResult,
    module_get_function: unsafe extern "C" fn(*mut Handle, Handle, *const c_char) -> CuResult,
    launch: unsafe extern "C" fn(
        Handle, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, Handle, *mut *mut c_void, *mut *mut c_void,
    ) -> CuResult,
    get_error_name: unsafe extern "C" fn(CuResult, *mut *const c_char) -> CuResult,
    get_error_string: unsafe extern "C" fn(CuResult, *mut *const c_char) -> CuResult,
}

unsafe impl Send for Driver {}
unsafe impl Sync for Driver {}

impl Driver {
    fn load() -> Result<Driver, String> {
        let name = if cfg!(windows) { "nvcuda.dll" } else { "libcuda.so.1" };
        unsafe {
            let lib = Library::new(name).map_err(|e| format!("cannot load {name}: {e}"))?;
            macro_rules! sym {
                ($n:literal) => {
                    *lib.get(concat!($n, "\0").as_bytes()).map_err(|e| format!("missing {}: {e}", $n))?
                };
            }
            Ok(Driver {
                cu_init: sym!("cuInit"),
                device_get_count: sym!("cuDeviceGetCount"),
                device_get: sym!("cuDeviceGet"),
                device_get_name: sym!("cuDeviceGetName"),
                primary_retain: sym!("cuDevicePrimaryCtxRetain"),
                ctx_set_current: sym!("cuCtxSetCurrent"),
                ctx_synchronize: sym!("cuCtxSynchronize"),
                mem_alloc: sym!("cuMemAlloc_v2"),
                mem_free: sym!("cuMemFree_v2"),
                htod: sym!("cuMemcpyHtoD_v2"),
                dtoh: sym!("cuMemcpyDtoH_v2"),
                dtod: sym!("cuMemcpyDtoD_v2"),
                mem_get_info: sym!("cuMemGetInfo_v2"),
                module_load_ex: sym!("cuModuleLoadDataEx"),
                module_get_function: sym!("cuModuleGetFunction"),
                launch: sym!("cuLaunchKernel"),
                get_error_name: sym!("cuGetErrorName"),
                get_error_string: sym!("cuGetErrorString"),
                _lib: lib,
            })
        }
    }

    fn describe(&self, code: CuResult) -> String {
        let get = |f: unsafe extern "C" fn(CuResult, *mut *const c_char) -> CuResult| unsafe {
            let mut p: *const c_char = std::ptr::null();
            if f(code, &mut p) == 0 && !p.is_null() { CStr::from_ptr(p).to_string_lossy().into_owned() } else { "?".into() }
        };
        format!("{} ({}) [code {code}]", get(self.get_error_name), get(self.get_error_string))
    }

    /// Maps a driver result to our error type, with context.
    fn check(&self, code: CuResult, what: &str) -> Result<(), E> {
        if code == 0 {
            return Ok(());
        }
        let status = if code == CUDA_ERROR_OUT_OF_MEMORY { STATUS_OUT_OF_MEMORY } else { STATUS_INTERNAL };
        err(status, format!("{what}: {}", self.describe(code)))
    }
}

/// Lazily loaded driver, `Err` if unavailable (no nvcuda.dll, cuInit failed...).
fn driver() -> Result<&'static Driver, &'static str> {
    static D: OnceLock<Result<Driver, String>> = OnceLock::new();
    D.get_or_init(|| {
        let d = Driver::load()?;
        let r = unsafe { (d.cu_init)(0) };
        if r != 0 {
            return Result::Err(format!("cuInit failed: {}", d.describe(r)));
        }
        Ok(d)
    })
    .as_ref()
    .map_err(|s| s.as_str())
}

// ---- cuBLAS (optional) -----------------------------------------------------------------------

struct Blas {
    _lib: Library,
    create: unsafe extern "C" fn(*mut Handle) -> c_int,
    sgemm: unsafe extern "C" fn(
        Handle, c_int, c_int, c_int, c_int, c_int, *const f32, CuPtr, c_int, CuPtr, c_int, *const f32, CuPtr, c_int,
    ) -> c_int,
}

unsafe impl Send for Blas {}
unsafe impl Sync for Blas {}

impl Blas {
    fn load() -> Result<Blas, String> {
        let mut candidates: Vec<String> = Vec::new();
        match std::env::var("PYTORCHES_CUBLAS") {
            Ok(v) if v == "0" => return Result::Err("disabled by PYTORCHES_CUBLAS=0".into()),
            Ok(v) if !v.is_empty() => candidates.push(v),
            _ => {}
        }
        let names: &[&str] = if cfg!(windows) {
            &["cublas64_13.dll", "cublas64_12.dll"]
        } else {
            &["libcublas.so.13", "libcublas.so.12"]
        };
        for n in names {
            candidates.push(n.to_string());
            if let Ok(cuda) = std::env::var("CUDA_PATH") {
                candidates.push(format!("{cuda}/bin/{n}"));
                candidates.push(format!("{cuda}/bin/x64/{n}"));
            }
        }
        let mut last = String::from("no candidates");
        for c in &candidates {
            unsafe {
                match Library::new(c) {
                    Ok(lib) => {
                        macro_rules! sym {
                            ($n:literal) => {
                                *lib.get(concat!($n, "\0").as_bytes()).map_err(|e| format!("{c}: missing {}: {e}", $n))?
                            };
                        }
                        return Ok(Blas { create: sym!("cublasCreate_v2"), sgemm: sym!("cublasSgemm_v2"), _lib: lib });
                    }
                    Err(e) => last = format!("cannot load {c}: {e}"),
                }
            }
        }
        Result::Err(last)
    }
}

fn blas() -> Option<&'static Blas> {
    static B: OnceLock<Option<Blas>> = OnceLock::new();
    B.get_or_init(|| match Blas::load() {
        Ok(b) => Some(b),
        Err(e) => {
            if std::env::var_os("PYTORCHES_CUDA_VERBOSE").is_some() {
                eprintln!("cuda plugin: cuBLAS unavailable ({e}); using built-in matmul");
            }
            None
        }
    })
    .as_ref()
}

/// A cuBLAS handle is not safe to use from several threads at once, hence the mutex.
struct BlasHandle(Mutex<usize>);

// ---- per-device state -----------------------------------------------------------------

#[derive(Default)]
struct Cache {
    free: HashMap<usize, Vec<CuPtr>>,
    sizes: HashMap<CuPtr, usize>,
    cached_bytes: usize,
}

struct Funcs {
    unary: Handle,
    binary: Handle,
    binary_vec: Handle,
    fill: Handle,
    randn: Handle,
    matmul: Handle,
    axpy: Handle,
    sum_rows: Handle,
    sum_cols: Handle,
    sum_parts: Handle,
}

struct Dev {
    dev: CuDevice,
    ctx: Handle,
    f: Funcs,
    blas: Option<BlasHandle>,
    cache: Mutex<Cache>,
}

unsafe impl Send for Dev {}
unsafe impl Sync for Dev {}

fn init_dev(drv: &Driver, index: u32) -> Result<Dev, E> {
    unsafe {
        let mut dev = 0;
        drv.check((drv.device_get)(&mut dev, index as c_int), "cuDeviceGet")?;
        let mut ctx = std::ptr::null_mut();
        drv.check((drv.primary_retain)(&mut ctx, dev), "cuDevicePrimaryCtxRetain")?;
        drv.check((drv.ctx_set_current)(ctx), "cuCtxSetCurrent")?;
        let ptx = CString::new(PTX).map_err(|_| E { status: STATUS_INTERNAL, msg: "PTX contains NUL".into() })?;
        let mut log = vec![0u8; 8192];
        let mut opts: [c_int; 2] = [5, 6]; // ERROR_LOG_BUFFER, ERROR_LOG_BUFFER_SIZE_BYTES
        let mut vals: [*mut c_void; 2] = [log.as_mut_ptr() as *mut c_void, log.len() as *mut c_void];
        let mut module = std::ptr::null_mut();
        let r = (drv.module_load_ex)(&mut module, ptx.as_ptr() as *const c_void, 2, opts.as_mut_ptr(), vals.as_mut_ptr());
        if r != 0 {
            let l = CStr::from_ptr(log.as_ptr() as *const c_char).to_string_lossy().into_owned();
            return err(STATUS_INTERNAL, format!("loading kernels PTX: {} {l}", drv.describe(r)));
        }
        let get = |name: &str| -> Result<Handle, E> {
            let c = CString::new(name).unwrap();
            let mut f = std::ptr::null_mut();
            drv.check((drv.module_get_function)(&mut f, module, c.as_ptr()), name)?;
            Ok(f)
        };
        let f = Funcs {
            unary: get("unary_k")?,
            binary: get("binary_k")?,
            binary_vec: get("binary_vec_k")?,
            fill: get("fill_k")?,
            randn: get("randn_k")?,
            matmul: get("matmul_k")?,
            axpy: get("axpy_k")?,
            sum_rows: get("sum_rows_k")?,
            sum_cols: get("sum_cols_k")?,
            sum_parts: get("sum_parts_k")?,
        };
        // cuBLAS handles bind to the current context, so create it now that ours is current.
        let blas = blas().and_then(|b| {
            let mut h: Handle = std::ptr::null_mut();
            (((b.create)(&mut h)) == 0).then(|| BlasHandle(Mutex::new(h as usize)))
        });
        Ok(Dev { dev, ctx, f, blas, cache: Mutex::new(Cache::default()) })
    }
}

/// Returns the device state (initializing on first use) with its context current on this thread.
fn enter(index: u32) -> Result<(&'static Driver, Arc<Dev>), E> {
    static DEVS: Mutex<Vec<(u32, Arc<Dev>)>> = Mutex::new(Vec::new());
    let drv = driver().map_err(|e| E { status: STATUS_INTERNAL, msg: e.to_string() })?;
    if index >= count() {
        return err(STATUS_INVALID_ARGUMENT, format!("no such device cuda:{index}"));
    }
    let mut devs = DEVS.lock().unwrap_or_else(|p| p.into_inner());
    let dev = match devs.iter().find(|(i, _)| *i == index) {
        Some((_, d)) => d.clone(),
        None => {
            let d = Arc::new(init_dev(drv, index)?);
            devs.push((index, d.clone()));
            d
        }
    };
    drop(devs);
    drv.check(unsafe { (drv.ctx_set_current)(dev.ctx) }, "cuCtxSetCurrent")?;
    Ok((drv, dev))
}

fn count() -> u32 {
    static N: OnceLock<u32> = OnceLock::new();
    *N.get_or_init(|| {
        catch_unwind(|| {
            let Ok(d) = driver() else { return 0 };
            let mut n = 0;
            if unsafe { (d.device_get_count)(&mut n) } != 0 || n < 0 { 0 } else { n as u32 }
        })
        .unwrap_or(0)
    })
}

// ---- caching allocator -------------------------------------------------------------------

fn bucket(bytes: usize) -> usize {
    const MIB: usize = 1 << 20;
    if bytes < MIB {
        bytes.max(256).next_power_of_two()
    } else {
        bytes.div_ceil(2 * MIB) * 2 * MIB
    }
}

impl Dev {
    fn flush(&self, drv: &Driver, c: &mut Cache) {
        for (size, ptrs) in c.free.drain() {
            for p in ptrs {
                unsafe { (drv.mem_free)(p) };
                c.sizes.remove(&p);
                c.cached_bytes -= size;
            }
        }
    }

    fn alloc(&self, drv: &Driver, bytes: usize) -> Result<CuPtr, E> {
        let size = bucket(bytes);
        let mut c = self.cache.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(p) = c.free.get_mut(&size).and_then(|v| v.pop()) {
            c.cached_bytes -= size;
            return Ok(p);
        }
        let mut p = 0;
        let mut r = unsafe { (drv.mem_alloc)(&mut p, size) };
        let mut got = size;
        if r == CUDA_ERROR_OUT_OF_MEMORY {
            self.flush(drv, &mut c);
            r = unsafe { (drv.mem_alloc)(&mut p, size) };
            if r == CUDA_ERROR_OUT_OF_MEMORY && bytes.div_ceil(256) * 256 < size {
                got = bytes.div_ceil(256) * 256;
                r = unsafe { (drv.mem_alloc)(&mut p, got) };
            }
        }
        drv.check(r, &format!("cuMemAlloc of {bytes} bytes"))?;
        c.sizes.insert(p, got);
        Ok(p)
    }

    fn release(&self, ptr: CuPtr) {
        let mut c = self.cache.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(&size) = c.sizes.get(&ptr) {
            c.free.entry(size).or_default().push(ptr);
            c.cached_bytes += size;
        }
    }
}

// ---- entry points: memory ------------------------------------------------------------------

unsafe extern "C" fn device_count() -> u32 {
    count()
}

unsafe extern "C" fn device_info(device: u32, out: *mut DeviceInfo) -> Status {
    wrap(|| {
        if out.is_null() {
            return err(STATUS_INVALID_ARGUMENT, "null out");
        }
        let (drv, dev) = enter(device)?;
        let mut info = DeviceInfo { name: [0; 64], kind: KIND_CUDA, total_memory: MEMORY_UNKNOWN, free_memory: MEMORY_UNKNOWN };
        unsafe {
            drv.check((drv.device_get_name)(info.name.as_mut_ptr(), 63, dev.dev), "cuDeviceGetName")?;
            let (mut free, mut total) = (0usize, 0usize);
            drv.check((drv.mem_get_info)(&mut free, &mut total), "cuMemGetInfo")?;
            let cached = dev.cache.lock().unwrap_or_else(|p| p.into_inner()).cached_bytes;
            info.total_memory = total as u64;
            info.free_memory = (free + cached).min(total) as u64;
            *out = info;
        }
        Ok(())
    })
}

unsafe extern "C" fn alloc(device: u32, bytes: usize, out: *mut *mut c_void) -> Status {
    wrap(|| {
        let (drv, dev) = enter(device)?;
        let p = dev.alloc(drv, bytes)?;
        unsafe { *out = p as usize as *mut c_void };
        Ok(())
    })
}

unsafe extern "C" fn free(device: u32, ptr: *mut c_void) {
    let _ = wrap(|| {
        if ptr.is_null() {
            return Ok(());
        }
        let (_, dev) = enter(device)?;
        dev.release(ptr as usize as CuPtr);
        Ok(())
    });
}

unsafe extern "C" fn copy_from_host(device: u32, dst: *mut c_void, src: *const c_void, bytes: usize) -> Status {
    wrap(|| {
        let (drv, _) = enter(device)?;
        if bytes == 0 {
            return Ok(());
        }
        drv.check(unsafe { (drv.htod)(dst as usize as CuPtr, src, bytes) }, "cuMemcpyHtoD")
    })
}

unsafe extern "C" fn copy_to_host(device: u32, dst: *mut c_void, src: *const c_void, bytes: usize) -> Status {
    wrap(|| {
        let (drv, _) = enter(device)?;
        if bytes == 0 {
            return Ok(());
        }
        drv.check(unsafe { (drv.dtoh)(dst, src as usize as CuPtr, bytes) }, "cuMemcpyDtoH")
    })
}

unsafe extern "C" fn copy_d2d(device: u32, dst: *mut c_void, src: *const c_void, bytes: usize) -> Status {
    wrap(|| {
        let (drv, _) = enter(device)?;
        if bytes == 0 {
            return Ok(());
        }
        drv.check(unsafe { (drv.dtod)(dst as usize as CuPtr, src as usize as CuPtr, bytes) }, "cuMemcpyDtoD")
    })
}

unsafe extern "C" fn synchronize(device: u32) -> Status {
    wrap(|| {
        let (drv, _) = enter(device)?;
        drv.check(unsafe { (drv.ctx_synchronize)() }, "cuCtxSynchronize")
    })
}

unsafe extern "C" fn last_error() -> *const c_char {
    LAST_ERROR.with(|e| {
        let e = e.borrow();
        if e.as_bytes().is_empty() { c"cuda plugin: no error".as_ptr() } else { e.as_ptr() }
    })
}

// ---- kernels ---------------------------------------------------------------------------------

/// Must match `Dims` in kernels.cu.
#[repr(C)]
#[derive(Clone, Copy)]
struct Dims {
    nd: u32,
    contig: u32,
    shape: [u64; 8],
    strides: [u64; 8],
}

/// Drops size-1 dims and merges adjacent dims that are jointly strided, so most layouts need <= 8 dims.
fn make_dims(shape: &[u64], strides: &[u64]) -> Result<Dims, E> {
    let mut v: Vec<(u64, u64)> = Vec::new();
    for (&s, &st) in shape.iter().zip(strides) {
        if s == 1 {
            continue;
        }
        if let Some(last) = v.last_mut() {
            if last.1 == st.wrapping_mul(s) {
                last.0 *= s;
                last.1 = st;
                continue;
            }
        }
        v.push((s, st));
    }
    if v.len() > 8 {
        return err(STATUS_UNSUPPORTED, format!("tensor needs {} dims after coalescing, max 8", v.len()));
    }
    let mut d = Dims { nd: v.len() as u32, contig: 0, shape: [1; 8], strides: [0; 8] };
    for (i, &(s, st)) in v.iter().enumerate() {
        d.shape[i] = s;
        d.strides[i] = st;
    }
    d.contig = (v.is_empty() || (v.len() == 1 && v[0].1 == 1)) as u32;
    Ok(d)
}

fn numel(shape: &[u64]) -> u64 {
    shape.iter().product()
}

fn p<T>(x: &mut T) -> *mut c_void {
    x as *mut T as *mut c_void
}

impl Driver {
    fn launch_k(&self, f: Handle, grid: (u32, u32), block: u32, params: &mut [*mut c_void], what: &str) -> Result<(), E> {
        if grid.0 == 0 || grid.1 == 0 || grid.1 > 65535 {
            return err(STATUS_UNSUPPORTED, format!("{what}: grid {grid:?} not launchable"));
        }
        let r = unsafe {
            (self.launch)(f, grid.0, grid.1, 1, block, 1, 1, 0, std::ptr::null_mut(), params.as_mut_ptr(), std::ptr::null_mut())
        };
        self.check(r, &format!("launch {what}"))
    }

    /// 1-D grid-stride launch over `n` elements.
    fn launch_1d(&self, f: Handle, n: u64, params: &mut [*mut c_void], what: &str) -> Result<(), E> {
        let blocks = n.div_ceil(256).min(1 << 18) as u32;
        self.launch_k(f, (blocks, 1), 256, params, what)
    }
}

fn dims_of<'a>(d: &TensorDesc) -> (&'a [u64], &'a [u64]) {
    unsafe {
        if d.ndim == 0 {
            (&[], &[])
        } else {
            (slice::from_raw_parts(d.shape, d.ndim as usize), slice::from_raw_parts(d.strides, d.ndim as usize))
        }
    }
}

/// Row-major `out[m,n] = op(a) x op(b)` over contiguous stored matrices; `ta`/`tb` mean the stored
/// matrix is the transpose of the operand (`a` is stored `[k,m]`, `b` is stored `[n,k]`).
struct Gemm {
    a: CuPtr,
    b: CuPtr,
    out: CuPtr,
    m: u64,
    n: u64,
    k: u64,
    ta: bool,
    tb: bool,
}

fn gemm(drv: &Driver, dev: &Dev, g: Gemm) -> Result<(), E> {
    let Gemm { a, b, out, m, n, k, ta, tb } = g;
    if let (Some(bl), Some(h)) = (blas(), dev.blas.as_ref()) {
        if m.max(n).max(k) <= i32::MAX as u64 {
            // Row-major C = A*B is column-major C^T = B^T * A^T, so the operands swap. A row-major
            // matrix read as column-major is already its transpose, so a stored operand needs the
            // transpose flag exactly when the caller asked for the transposed one.
            let (alpha, beta) = (1f32, 0f32);
            let (op_b, op_a) = (tb as c_int, ta as c_int);
            let ldb = if tb { k } else { n } as c_int;
            let lda = if ta { m } else { k } as c_int;
            let r = {
                let hd = h.0.lock().unwrap_or_else(|p| p.into_inner());
                unsafe {
                    (bl.sgemm)(
                        *hd as Handle, op_b, op_a, n as c_int, m as c_int, k as c_int, &alpha, b, ldb, a, lda, &beta, out,
                        n as c_int,
                    )
                }
            };
            return if r == 0 { Ok(()) } else { err(STATUS_INTERNAL, format!("cublasSgemm failed with status {r}")) };
        }
    }
    // Built-in kernel: contiguous row-major only, so materialize any transposed operand first.
    let mut scratch: Vec<CuPtr> = Vec::new();
    let mut transposed = |src: CuPtr, rows: u64, cols: u64| -> Result<CuPtr, E> {
        // `src` is stored [rows, cols]; the copy is its transpose [cols, rows].
        let t = dev.alloc(drv, (rows * cols * 4) as usize)?;
        scratch.push(t);
        let mut d = make_dims(&[cols, rows], &[1, cols])?;
        let (mut inp, mut dst, mut nn, mut code) = (src, t, rows * cols, 7 as c_int);
        drv.launch_1d(dev.f.unary, nn, &mut [p(&mut code), p(&mut inp), p(&mut dst), p(&mut nn), p(&mut d)], "transpose")?;
        Ok(t)
    };
    let res = (|| {
        let mut a = if ta { transposed(a, k, m)? } else { a };
        let mut b = if tb { transposed(b, n, k)? } else { b };
        let (mut mm, mut kk, mut nn, mut out) = (m, k, n, out);
        let (gx, gy) = (n.div_ceil(128) as u32, m.div_ceil(128) as u32);
        if gy > 65535 || gx == 0 {
            return err(STATUS_UNSUPPORTED, "matmul too large");
        }
        // 16x16 thread block: launch with block (16,16,1) via grid/block override.
        let r = unsafe {
            let mut params = [p(&mut a), p(&mut b), p(&mut out), p(&mut mm), p(&mut nn), p(&mut kk)];
            (drv.launch)(dev.f.matmul, gx, gy, 1, 16, 16, 1, 0, std::ptr::null_mut(), params.as_mut_ptr(), std::ptr::null_mut())
        };
        drv.check(r, "launch matmul")
    })();
    // Stream ordering makes it safe to recycle the scratch right away.
    for t in scratch {
        dev.release(t);
    }
    res
}

fn run(device: u32, op_code: u32, attrs: &OpAttrs, ins: &[TensorDesc], outs: &[TensorDesc]) -> Result<(), E> {
    use op::*;
    let (drv, dev) = enter(device)?;
    if outs.len() != 1 {
        return err(STATUS_INVALID_ARGUMENT, "expected exactly one output");
    }
    let need = match op_code {
        NEG | EXP | LOG | RELU | TANH | STEP | COPY => 1,
        ADD | SUB | MUL | DIV | MATMUL | MATMUL_T | AXPY => 2,
        SUM_AXIS => 1,
        FILL | RAND_NORMAL => 0,
        _ => return err(STATUS_UNSUPPORTED, format!("unsupported op {op_code}")),
    };
    if ins.len() != need {
        return err(STATUS_INVALID_ARGUMENT, format!("op {op_code} needs {need} inputs, got {}", ins.len()));
    }
    if ins.iter().chain(outs).any(|d| d.dtype != DTYPE_F32) {
        return err(STATUS_UNSUPPORTED, "only f32 is supported");
    }
    let (oshape, _) = dims_of(&outs[0]);
    let n = numel(oshape);
    let mut out = outs[0].data as usize as CuPtr;
    if n == 0 {
        return Ok(());
    }
    let mut nn = n;
    let f = &dev.f;

    match op_code {
        FILL => {
            let mut v = f32::from_bits(attrs.ints[0] as u32);
            drv.launch_1d(f.fill, n, &mut [p(&mut out), p(&mut nn), p(&mut v)], "fill")
        }
        RAND_NORMAL => {
            let mut seed = attrs.ints[0] as u64;
            drv.launch_1d(f.randn, n, &mut [p(&mut out), p(&mut nn), p(&mut seed)], "randn")
        }
        NEG | EXP | LOG | RELU | TANH | STEP | COPY => {
            let (s, st) = dims_of(&ins[0]);
            if numel(s) != n {
                return err(STATUS_INVALID_ARGUMENT, "input shape does not match output");
            }
            let mut d = make_dims(s, st)?;
            let mut inp = ins[0].data as usize as CuPtr;
            let mut code: c_int = if op_code == COPY { 7 } else { op_code as c_int };
            drv.launch_1d(f.unary, n, &mut [p(&mut code), p(&mut inp), p(&mut out), p(&mut nn), p(&mut d)], "unary")
        }
        ADD | SUB | MUL | DIV => {
            let (sa, sta) = dims_of(&ins[0]);
            let (sb, stb) = dims_of(&ins[1]);
            if numel(sa) != n || numel(sb) != n {
                return err(STATUS_INVALID_ARGUMENT, "input shapes do not match output");
            }
            let (mut da, mut db) = (make_dims(sa, sta)?, make_dims(sb, stb)?);
            let (mut a, mut b) = (ins[0].data as usize as CuPtr, ins[1].data as usize as CuPtr);
            let mut code: c_int = (op_code - ADD) as c_int;
            // Contiguous same-shape operands with aligned pointers take the 128-bit path.
            if da.contig == 1 && db.contig == 1 && n % 4 == 0 && (a | b | out) % 16 == 0 {
                let mut n4 = n / 4;
                return drv.launch_1d(f.binary_vec, n4, &mut [p(&mut code), p(&mut a), p(&mut b), p(&mut out), p(&mut n4)], "binary_vec");
            }
            drv.launch_1d(
                f.binary,
                n,
                &mut [p(&mut code), p(&mut a), p(&mut b), p(&mut out), p(&mut nn), p(&mut da), p(&mut db)],
                "binary",
            )
        }
        MATMUL | MATMUL_T => {
            let flags = if op_code == MATMUL_T { attrs.ints[0] } else { 0 };
            let (ta, tb) = (flags & 1 != 0, flags & 2 != 0);
            let (sa, _) = dims_of(&ins[0]);
            let (sb, _) = dims_of(&ins[1]);
            if sa.len() != 2 || sb.len() != 2 {
                return err(STATUS_INVALID_ARGUMENT, "matmul needs 2-D inputs");
            }
            let (m, k) = if ta { (sa[1], sa[0]) } else { (sa[0], sa[1]) };
            let (kb, nc) = if tb { (sb[1], sb[0]) } else { (sb[0], sb[1]) };
            if k != kb || n != m * nc {
                return err(STATUS_INVALID_ARGUMENT, "matmul shape mismatch");
            }
            let (a, b) = (ins[0].data as usize as CuPtr, ins[1].data as usize as CuPtr);
            gemm(drv, &dev, Gemm { a, b, out, m, n: nc, k, ta, tb })
        }
        AXPY => {
            let (sa, _) = dims_of(&ins[0]);
            let (sb, _) = dims_of(&ins[1]);
            if numel(sa) != n || numel(sb) != n {
                return err(STATUS_INVALID_ARGUMENT, "axpy input shapes do not match output");
            }
            let (mut a, mut b) = (ins[0].data as usize as CuPtr, ins[1].data as usize as CuPtr);
            let mut alpha = f32::from_bits(attrs.ints[0] as u32);
            drv.launch_1d(f.axpy, n, &mut [p(&mut a), p(&mut b), p(&mut out), p(&mut nn), p(&mut alpha)], "axpy")
        }
        SUM_AXIS => {
            let (shape, _) = dims_of(&ins[0]);
            let axis = attrs.ints[0];
            if axis < 0 || axis as usize >= shape.len() {
                return err(STATUS_INVALID_ARGUMENT, "sum axis out of range");
            }
            let axis = axis as usize;
            let (mut outer, mut len, mut inner) = (numel(&shape[..axis]), shape[axis], numel(&shape[axis + 1..]));
            let total = outer * inner;
            if total != n {
                return err(STATUS_INVALID_ARGUMENT, "sum output shape mismatch");
            }
            let mut x = ins[0].data as usize as CuPtr;
            if len == 0 {
                let mut z = 0f32;
                return drv.launch_1d(f.fill, n, &mut [p(&mut out), p(&mut nn), p(&mut z)], "fill(sum of empty)");
            }
            // Pick the number of chunks the reduced axis is split into (parts > 1 needs scratch).
            let (parts, chunk) = if inner == 1 {
                let np = if outer >= 256 { 1 } else { (len / 4096).clamp(1, 1024) };
                (np, len.div_ceil(np))
            } else {
                let np = if total >= 32768 || len <= 64 { 1 } else { (131072 / total).clamp(1, len.div_ceil(32)).min(4096) };
                (np, len.div_ceil(np))
            };
            // Rows that are a multiple of 4 floats, 16-byte aligned, can be read as float4: keep every
            // chunk a multiple of 4 too so each chunk start stays aligned.
            let mut vec: c_int = (inner == 1 && len % 4 == 0 && x % 16 == 0) as c_int;
            let chunk = if vec == 1 { chunk.next_multiple_of(4) } else { chunk };
            let parts = len.div_ceil(chunk).min(parts).max(1);
            let mut chunk_v = chunk;
            let scratch = if parts > 1 { Some(dev.alloc(drv, (parts * total * 4) as usize)?) } else { None };
            let mut dst = scratch.unwrap_or(out);
            let res = (|| {
                if inner == 1 {
                    drv.launch_k(
                        f.sum_rows,
                        (outer as u32, parts as u32),
                        256,
                        &mut [p(&mut x), p(&mut dst), p(&mut outer), p(&mut len), p(&mut chunk_v), p(&mut vec)],
                        "sum_rows",
                    )
                } else {
                    let gx = total.div_ceil(256).min(1 << 18) as u32;
                    drv.launch_k(
                        f.sum_cols,
                        (gx, parts as u32),
                        256,
                        &mut [p(&mut x), p(&mut dst), p(&mut outer), p(&mut len), p(&mut inner), p(&mut chunk_v)],
                        "sum_cols",
                    )
                }?;
                if scratch.is_some() {
                    let mut tot = total;
                    let mut np = parts;
                    drv.launch_1d(f.sum_parts, total, &mut [p(&mut dst), p(&mut out), p(&mut tot), p(&mut np)], "sum_parts")?;
                }
                Ok(())
            })();
            // Stream ordering makes it safe to recycle the scratch immediately.
            if let Some(s) = scratch {
                dev.release(s);
            }
            res
        }
        _ => err(STATUS_UNSUPPORTED, format!("unsupported op {op_code}")),
    }
}

unsafe extern "C" fn supports_op(code: u32) -> u32 {
    use op::*;
    matches!(
        code,
        NEG | EXP | LOG | RELU | TANH | STEP | ADD | SUB | MUL | DIV | MATMUL | MATMUL_T | AXPY | SUM_AXIS | COPY | FILL | RAND_NORMAL
    ) as u32
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
    wrap(|| unsafe {
        let attrs = if attrs.is_null() { OpAttrs::default() } else { *attrs };
        let ins: &[TensorDesc] = if n_inputs == 0 { &[] } else { slice::from_raw_parts(inputs, n_inputs as usize) };
        let outs: &[TensorDesc] = if n_outputs == 0 { &[] } else { slice::from_raw_parts(outputs, n_outputs as usize) };
        run(device, op_code, &attrs, ins, outs)
    })
}

static VTABLE: PluginVTable = PluginVTable {
    abi_version: ABI_VERSION,
    plugin_version: 1,
    name: c"cuda".as_ptr(),
    device_count,
    device_info,
    alloc,
    free,
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
