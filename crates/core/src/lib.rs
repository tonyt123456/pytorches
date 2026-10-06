//! PyTorches core: tensors and reverse-mode autograd.
//!
//! The core contains no device code. Every tensor lives in a buffer owned by a plugin
//! (see [`plugin`]) and every op is dispatched to that plugin through the C ABI in
//! `pytorches-plugin-abi`.
//!
//! Current scope: `f32`, contiguous tensors. Broadcasting and transposes are expressed
//! as strided operands at dispatch time rather than as stored views.

pub mod plan;
pub mod plugin;

use plugin::Plugin;
use pytorches_plugin_abi::{self as abi, OpAttrs, TensorDesc, op};
use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::fmt;
use std::panic::{self, AssertUnwindSafe};
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Once};

fn next_id() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

// ---- errors ---------------------------------------------------------------------

/// Errors raised by the core.
///
/// Internally they travel as typed panic payloads, so op signatures stay simple. Callers that
/// want to handle them (the Python bindings, embedders) wrap work in [`try_run`], which turns any
/// panic into an `Error`: typed errors keep their kind, other panics (shape or device mismatches
/// raised with `assert!`) become [`Error::Invalid`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// A device ran out of memory. The tensor was not created; other devices may still work.
    OutOfMemory(String),
    /// The caller passed something invalid (shape/device mismatch, unsupported op, bad argument).
    Invalid(String),
    /// A plugin reported an unexpected failure.
    Backend(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::OutOfMemory(m) | Error::Invalid(m) | Error::Backend(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for Error {}

thread_local! {
    static GUARD_DEPTH: Cell<u32> = const { Cell::new(0) };
}

fn install_quiet_hook() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let prev = panic::take_hook();
        // Inside `try_run` the panic is reported through the returned `Error`, so don't also
        // print it to stderr.
        panic::set_hook(Box::new(move |info| {
            if GUARD_DEPTH.with(Cell::get) == 0 {
                prev(info);
            }
        }));
    });
}

/// Runs `f`, converting any panic raised inside it into an [`Error`].
pub fn try_run<T>(f: impl FnOnce() -> T) -> Result<T, Error> {
    install_quiet_hook();
    GUARD_DEPTH.with(|d| d.set(d.get() + 1));
    let result = panic::catch_unwind(AssertUnwindSafe(f));
    GUARD_DEPTH.with(|d| d.set(d.get() - 1));
    result.map_err(|payload| match payload.downcast::<Error>() {
        Ok(e) => *e,
        Err(payload) => {
            let msg = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "internal error".into());
            Error::Invalid(msg)
        }
    })
}

fn raise(e: Error) -> ! {
    panic::resume_unwind(Box::new(e))
}

/// Raises the right [`Error`] kind if a plugin call did not return `STATUS_OK`.
fn check(status: abi::Status, plugin: &Plugin, what: impl FnOnce() -> String) {
    if status == abi::STATUS_OK {
        return;
    }
    let msg = format!("{}: {}", what(), plugin.last_error());
    raise(match status {
        abi::STATUS_OUT_OF_MEMORY => Error::OutOfMemory(msg),
        abi::STATUS_INVALID_ARGUMENT | abi::STATUS_UNSUPPORTED => Error::Invalid(msg),
        _ => Error::Backend(msg),
    })
}

// ---- devices ------------------------------------------------------------------

#[derive(Clone)]
pub struct Device {
    plugin: Arc<Plugin>,
    index: u32,
}

#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub name: String,
    pub kind: u32,
    pub total_memory: Option<u64>,
    pub free_memory: Option<u64>,
}

impl Device {
    /// Parses `"cuda:0"`, `"cpu"` (index defaults to 0).
    pub fn parse(s: &str) -> Result<Device, String> {
        let (name, index) = match s.split_once(':') {
            Some((n, i)) => (n, i.parse::<u32>().map_err(|_| format!("bad device index in '{s}'"))?),
            None => (s, 0),
        };
        let plugin = plugin::find_plugin(name).ok_or_else(|| {
            let loaded: Vec<String> = plugin::plugins().iter().map(|p| p.name.clone()).collect();
            format!("no plugin '{name}' loaded (loaded: {loaded:?})")
        })?;
        if index >= plugin.device_count() {
            return Err(format!("plugin '{name}' has {} device(s), asked for index {index}", plugin.device_count()));
        }
        Ok(Device { plugin, index })
    }

    /// `cpu:0` if the cpu plugin is loaded, otherwise the first loaded plugin's first device.
    pub fn default_device() -> Device {
        let plugin = plugin::find_plugin("cpu")
            .or_else(|| plugin::plugins().into_iter().next())
            .expect("no plugins loaded; set PYTORCHES_PLUGIN_DIR or call plugin::load_plugin_dir");
        Device { plugin, index: 0 }
    }

    /// Every device of every loaded plugin.
    pub fn all() -> Vec<Device> {
        plugin::plugins()
            .into_iter()
            .flat_map(|p| (0..p.device_count()).map(move |i| Device { plugin: p.clone(), index: i }))
            .collect()
    }

    pub fn info(&self) -> DeviceInfo {
        let mut raw = abi::DeviceInfo { name: [0; 64], kind: abi::KIND_OTHER, total_memory: 0, free_memory: 0 };
        let status = unsafe { (self.plugin.vt.device_info)(self.index, &mut raw) };
        assert_eq!(status, abi::STATUS_OK, "device_info failed: {}", self.plugin.last_error());
        let name = unsafe { std::ffi::CStr::from_ptr(raw.name.as_ptr()) }.to_string_lossy().into_owned();
        let mem = |v: u64| (v != abi::MEMORY_UNKNOWN).then_some(v);
        DeviceInfo { name, kind: raw.kind, total_memory: mem(raw.total_memory), free_memory: mem(raw.free_memory) }
    }

    pub fn synchronize(&self) {
        unsafe { (self.plugin.vt.synchronize)(self.index) };
    }
}

impl PartialEq for Device {
    fn eq(&self, o: &Device) -> bool {
        Arc::ptr_eq(&self.plugin, &o.plugin) && self.index == o.index
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.plugin.name, self.index)
    }
}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Device({self})")
    }
}

// ---- device buffers -------------------------------------------------------------

/// Owns a plugin allocation; freed through the plugin on drop.
struct Buffer {
    device: Device,
    ptr: *mut c_void,
    /// Set for memory owned by someone else (e.g. a DLPack producer); runs instead of the
    /// plugin's `free` when the last reference to the buffer is dropped.
    foreign: Option<Box<dyn FnOnce() + Send>>,
}

// The pointer is opaque and only ever used via thread-safe plugin entry points.
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    fn alloc(device: &Device, numel: usize) -> Buffer {
        let mut ptr = std::ptr::null_mut();
        let bytes = (numel * 4).max(4);
        let status = unsafe { (device.plugin.vt.alloc)(device.index, bytes, &mut ptr) };
        check(status, &device.plugin, || format!("alloc of {bytes} bytes on {device} failed"));
        Buffer { device: device.clone(), ptr, foreign: None }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        match self.foreign.take() {
            Some(release) => release(),
            None => unsafe { (self.device.plugin.vt.free)(self.device.index, self.ptr) },
        }
    }
}

/// Host memory of a CPU tensor, plus a guard that keeps it alive. See [`Tensor::host_export`].
pub struct HostExport {
    pub ptr: *mut c_void,
    /// Drop this when the consumer is done with `ptr`.
    pub keepalive: Box<dyn std::any::Any + Send + Sync>,
}

// ---- tensor ---------------------------------------------------------------------

type BackwardFn = Box<dyn Fn(&Tensor) -> Vec<Option<Tensor>> + Send + Sync>;

struct Node {
    inputs: Vec<Tensor>,
    backward: BackwardFn,
}

struct Inner {
    id: u64,
    data: Arc<Buffer>,
    shape: Vec<usize>,
    requires_grad: bool,
    grad: Mutex<Option<Tensor>>,
    grad_fn: Option<Node>,
}

#[derive(Clone)]
pub struct Tensor(Arc<Inner>);

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor(shape={:?}, device={}, data={:?})", self.shape(), self.device(), self.to_vec())
    }
}

fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];
    let mut acc = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        acc *= shape[i];
    }
    strides
}

/// NumPy-style broadcast of two shapes; `None` if incompatible.
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let n = a.len().max(b.len());
    let (pa, pb) = (n - a.len(), n - b.len());
    (0..n)
        .map(|i| {
            let da = if i < pa { 1 } else { a[i - pa] };
            let db = if i < pb { 1 } else { b[i - pb] };
            match (da, db) {
                (x, y) if x == y => Some(x),
                (1, y) => Some(y),
                (x, 1) => Some(x),
                _ => None,
            }
        })
        .collect()
}

/// Strides to read contiguous `shape` as if broadcast to `out_shape` (0 on broadcast dims).
fn broadcast_strides(shape: &[usize], out_shape: &[usize]) -> Vec<usize> {
    let pad = out_shape.len() - shape.len();
    let base = contiguous_strides(shape);
    let mut strides = vec![0; out_shape.len()];
    for i in 0..shape.len() {
        strides[pad + i] = if shape[i] == 1 { 0 } else { base[i] };
    }
    strides
}

/// An op argument: a tensor read through an explicit shape/strides.
struct Operand<'a> {
    t: &'a Tensor,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl<'a> Operand<'a> {
    fn contiguous(t: &'a Tensor) -> Self {
        Operand { t, shape: t.shape().to_vec(), strides: contiguous_strides(t.shape()) }
    }

    fn broadcast(t: &'a Tensor, out_shape: &[usize]) -> Self {
        Operand { t, shape: out_shape.to_vec(), strides: broadcast_strides(t.shape(), out_shape) }
    }
}

fn run_op(device: &Device, code: u32, ints: [i64; 4], ins: &[Operand], out_shape: &[usize]) -> Tensor {
    let out_buf = Arc::new(Buffer::alloc(device, out_shape.iter().product()));
    run_op_into(device, code, ints, ins, &out_buf, out_shape);
    Tensor::build(out_buf, out_shape.to_vec(), false, None)
}

/// Runs one op, writing the (contiguous) result into an existing buffer.
fn run_op_into(device: &Device, code: u32, ints: [i64; 4], ins: &[Operand], out_buf: &Arc<Buffer>, out_shape: &[usize]) {
    let vt = device.plugin.vt;
    assert!(
        unsafe { (vt.supports_op)(code) } != 0,
        "plugin '{}' does not implement op {code}",
        device.plugin.name
    );
    for o in ins {
        assert!(&o.t.device() == device, "tensors on different devices: {} and {device}", o.t.device());
    }
    assert!(&out_buf.device == device, "output buffer is on {}, op runs on {device}", out_buf.device);

    let to_u64 = |v: &[usize]| v.iter().map(|&x| x as u64).collect::<Vec<u64>>();
    // Keep the dimension arrays alive until after the call.
    let in_dims: Vec<(Vec<u64>, Vec<u64>)> = ins.iter().map(|o| (to_u64(&o.shape), to_u64(&o.strides))).collect();
    let out_dims = (to_u64(out_shape), to_u64(&contiguous_strides(out_shape)));

    let desc = |data: *mut c_void, d: &(Vec<u64>, Vec<u64>)| TensorDesc {
        data,
        dtype: abi::DTYPE_F32,
        ndim: d.0.len() as u32,
        shape: d.0.as_ptr(),
        strides: d.1.as_ptr(),
    };
    let in_descs: Vec<TensorDesc> = ins.iter().zip(&in_dims).map(|(o, d)| desc(o.t.0.data.ptr, d)).collect();
    let out_desc = desc(out_buf.ptr, &out_dims);
    let attrs = OpAttrs { ints };

    let status = unsafe {
        (vt.execute)(device.index, code, &attrs, in_descs.as_ptr(), in_descs.len() as u32, &out_desc, 1)
    };
    check(status, &device.plugin, || format!("op {code} failed on {device}"));
}

impl Tensor {
    fn build(data: Arc<Buffer>, shape: Vec<usize>, requires_grad: bool, grad_fn: Option<Node>) -> Self {
        Tensor(Arc::new(Inner { id: next_id(), data, shape, requires_grad, grad: Mutex::new(None), grad_fn }))
    }

    // ---- construction ----------------------------------------------------

    pub fn from_vec_on(data: Vec<f32>, shape: Vec<usize>, device: &Device) -> Self {
        assert_eq!(data.len(), shape.iter().product::<usize>(), "data length does not match shape {shape:?}");
        let buf = Buffer::alloc(device, data.len());
        let status = unsafe {
            (device.plugin.vt.copy_from_host)(device.index, buf.ptr, data.as_ptr() as *const c_void, data.len() * 4)
        };
        check(status, &device.plugin, || "host->device copy failed".to_string());
        Self::build(Arc::new(buf), shape, false, None)
    }

    /// On the default device.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self::from_vec_on(data, shape, &Device::default_device())
    }

    pub fn scalar_on(v: f32, device: &Device) -> Self {
        Self::full_on(&[], v, device)
    }

    /// Constant-filled tensor, created directly on the device (no host staging).
    pub fn full_on(shape: &[usize], value: f32, device: &Device) -> Self {
        run_op(device, op::FILL, [value.to_bits() as i64, 0, 0, 0], &[], shape)
    }

    pub fn ones_on(shape: &[usize], device: &Device) -> Self {
        Self::full_on(shape, 1.0, device)
    }

    pub fn zeros_on(shape: &[usize], device: &Device) -> Self {
        Self::full_on(shape, 0.0, device)
    }

    /// N(0,1) samples generated on the device. Same seed gives the same values on every device
    /// (up to float rounding in `ln`/`cos`).
    pub fn randn_on(shape: &[usize], seed: u64, device: &Device) -> Self {
        run_op(device, op::RAND_NORMAL, [seed as i64, 0, 0, 0], &[], shape)
    }

    /// Wraps host `f32` memory owned by someone else (zero copy) as a CPU tensor.
    ///
    /// `release` runs when the last reference to the tensor's storage is dropped. If the pointer is
    /// null or not 4-byte aligned, or the tensor is empty, the data is copied instead and `release`
    /// runs immediately.
    ///
    /// # Safety
    /// `ptr` must point to `shape.iter().product()` valid, contiguous `f32`s that stay valid and are
    /// not freed until `release` runs.
    pub unsafe fn from_host_borrowed(
        ptr: *mut f32,
        shape: Vec<usize>,
        release: impl FnOnce() + Send + 'static,
    ) -> Tensor {
        let device = Device::parse("cpu").unwrap_or_else(|e| raise(Error::Invalid(e)));
        let numel: usize = shape.iter().product();
        if ptr.is_null() || numel == 0 || (ptr as usize) % std::mem::align_of::<f32>() != 0 {
            let data = if numel == 0 || ptr.is_null() {
                Vec::new()
            } else {
                unsafe { std::slice::from_raw_parts(ptr, numel) }.to_vec()
            };
            release();
            return Tensor::from_vec_on(data, shape, &device);
        }
        let buf = Buffer { device, ptr: ptr as *mut c_void, foreign: Some(Box::new(release)) };
        Self::build(Arc::new(buf), shape, false, None)
    }

    /// For CPU tensors: the host pointer of the (contiguous, row-major) data plus a guard keeping
    /// it alive, enabling zero-copy hand-off. `None` for tensors on other devices.
    pub fn host_export(&self) -> Option<HostExport> {
        let buf = &self.0.data;
        if buf.device.info().kind != abi::KIND_CPU {
            return None;
        }
        Some(HostExport { ptr: buf.ptr, keepalive: Box::new(buf.clone()) })
    }

    /// A leaf tensor sharing this tensor's storage with the given `requires_grad`.
    pub fn requires_grad_(&self, flag: bool) -> Self {
        Self::build(self.0.data.clone(), self.0.shape.clone(), flag, None)
    }

    pub fn detach(&self) -> Self {
        self.requires_grad_(false)
    }

    /// Wraps an op result, recording the graph only if some input needs grad.
    fn attach(
        self,
        inputs: Vec<Tensor>,
        backward: impl Fn(&Tensor) -> Vec<Option<Tensor>> + Send + Sync + 'static,
    ) -> Tensor {
        if !inputs.iter().any(Tensor::requires_grad) {
            return self;
        }
        let node = Node { inputs, backward: Box::new(backward) };
        Self::build(self.0.data.clone(), self.0.shape.clone(), true, Some(node))
    }

    // ---- accessors ---------------------------------------------------------

    pub fn shape(&self) -> &[usize] {
        &self.0.shape
    }

    pub fn numel(&self) -> usize {
        self.0.shape.iter().product()
    }

    pub fn device(&self) -> Device {
        self.0.data.device.clone()
    }

    /// Size of the tensor's storage in bytes.
    pub fn nbytes(&self) -> usize {
        self.numel() * 4
    }

    /// In-place overwrite with `src` (same shape; `src` is moved to this device if needed).
    /// Not tracked by autograd. Storage shared with `detach()` copies sees the change.
    pub fn copy_(&self, src: &Tensor) {
        assert_eq!(self.shape(), src.shape(), "copy_ shape mismatch: {:?} vs {:?}", self.shape(), src.shape());
        let dev = self.device();
        let src = src.to(&dev);
        run_op_into(&dev, op::COPY, [0; 4], &[Operand::contiguous(&src)], &self.0.data, self.shape());
    }

    /// Copies the data to the host.
    pub fn to_vec(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; self.numel()];
        let dev = &self.0.data.device;
        let status = unsafe {
            (dev.plugin.vt.copy_to_host)(dev.index, out.as_mut_ptr() as *mut c_void, self.0.data.ptr, out.len() * 4)
        };
        check(status, &dev.plugin, || "device->host copy failed".to_string());
        out
    }

    pub fn requires_grad(&self) -> bool {
        self.0.requires_grad
    }

    pub fn grad(&self) -> Option<Tensor> {
        self.0.grad.lock().unwrap().clone()
    }

    pub fn zero_grad(&self) {
        *self.0.grad.lock().unwrap() = None;
    }

    pub fn item(&self) -> f32 {
        assert_eq!(self.numel(), 1, "item() needs a single-element tensor");
        self.to_vec()[0]
    }

    /// Moves the tensor to `device` (staged through host memory if it differs). Differentiable.
    pub fn to(&self, device: &Device) -> Tensor {
        if &self.device() == device {
            return self.clone();
        }
        let moved = Tensor::from_vec_on(self.to_vec(), self.shape().to_vec(), device);
        let src = self.device();
        moved.attach(vec![self.clone()], move |g| vec![Some(g.to(&src))])
    }

    // ---- raw (graph-free) kernels ------------------------------------------

    fn raw_unary(&self, code: u32) -> Tensor {
        run_op(&self.device(), code, [0; 4], &[Operand::contiguous(self)], self.shape())
    }

    fn raw_binary(&self, o: &Tensor, code: u32) -> Tensor {
        let out_shape = broadcast_shapes(self.shape(), o.shape())
            .unwrap_or_else(|| panic!("shapes {:?} and {:?} are not broadcastable", self.shape(), o.shape()));
        let operands = [Operand::broadcast(self, &out_shape), Operand::broadcast(o, &out_shape)];
        run_op(&self.device(), code, [0; 4], &operands, &out_shape)
    }

    fn raw_sum_axis(&self, axis: usize) -> Tensor {
        let mut shape = self.shape().to_vec();
        shape.remove(axis);
        run_op(&self.device(), op::SUM_AXIS, [axis as i64, 0, 0, 0], &[Operand::contiguous(self)], &shape)
    }

    /// Graph-free reshape sharing storage.
    fn raw_reshape(&self, shape: &[usize]) -> Tensor {
        assert_eq!(shape.iter().product::<usize>(), self.numel(), "cannot reshape {:?} to {shape:?}", self.shape());
        Tensor::build(self.0.data.clone(), shape.to_vec(), false, None)
    }

    /// Graph-free contiguous transpose of a 2-D tensor (a strided COPY).
    fn raw_t(&self) -> Tensor {
        assert_eq!(self.shape().len(), 2, "t() needs a 2-D tensor");
        let (r, c) = (self.shape()[0], self.shape()[1]);
        let operand = Operand { t: self, shape: vec![c, r], strides: vec![1, c] };
        run_op(&self.device(), op::COPY, [0; 4], &[operand], &[c, r])
    }

    /// Reduce a broadcast result back to `target` (inverse of broadcasting).
    fn unbroadcast(&self, target: &[usize]) -> Tensor {
        let mut g = self.clone();
        while g.shape().len() > target.len() {
            g = g.raw_sum_axis(0);
        }
        for (axis, &dim) in target.iter().enumerate() {
            if dim == 1 && g.shape()[axis] != 1 {
                let mut keep = g.shape().to_vec();
                keep[axis] = 1;
                g = g.raw_sum_axis(axis).raw_reshape(&keep);
            }
        }
        g
    }

    // ---- differentiable ops ------------------------------------------------

    pub fn add(&self, o: &Tensor) -> Tensor {
        let (sa, sb) = (self.shape().to_vec(), o.shape().to_vec());
        self.raw_binary(o, op::ADD).attach(vec![self.clone(), o.clone()], move |g| {
            vec![Some(g.unbroadcast(&sa)), Some(g.unbroadcast(&sb))]
        })
    }

    pub fn sub(&self, o: &Tensor) -> Tensor {
        let (sa, sb) = (self.shape().to_vec(), o.shape().to_vec());
        self.raw_binary(o, op::SUB).attach(vec![self.clone(), o.clone()], move |g| {
            vec![Some(g.unbroadcast(&sa)), Some(g.neg().unbroadcast(&sb))]
        })
    }

    pub fn mul(&self, o: &Tensor) -> Tensor {
        let (a, b) = (self.detach(), o.detach());
        self.raw_binary(o, op::MUL).attach(vec![self.clone(), o.clone()], move |g| {
            vec![Some(g.mul(&b).unbroadcast(a.shape())), Some(g.mul(&a).unbroadcast(b.shape()))]
        })
    }

    pub fn div(&self, o: &Tensor) -> Tensor {
        let (a, b) = (self.detach(), o.detach());
        self.raw_binary(o, op::DIV).attach(vec![self.clone(), o.clone()], move |g| {
            let gb = g.mul(&a).div(&b.mul(&b)).neg();
            vec![Some(g.div(&b).unbroadcast(a.shape())), Some(gb.unbroadcast(b.shape()))]
        })
    }

    pub fn neg(&self) -> Tensor {
        self.raw_unary(op::NEG).attach(vec![self.clone()], |g| vec![Some(g.neg())])
    }

    pub fn exp(&self) -> Tensor {
        let out = self.raw_unary(op::EXP);
        let o = out.clone();
        out.attach(vec![self.clone()], move |g| vec![Some(g.mul(&o))])
    }

    pub fn log(&self) -> Tensor {
        let x = self.detach();
        self.raw_unary(op::LOG).attach(vec![self.clone()], move |g| vec![Some(g.div(&x))])
    }

    pub fn relu(&self) -> Tensor {
        let x = self.detach();
        self.raw_unary(op::RELU)
            .attach(vec![self.clone()], move |g| vec![Some(g.mul(&x.raw_unary(op::STEP)))])
    }

    pub fn tanh(&self) -> Tensor {
        let out = self.raw_unary(op::TANH);
        let o = out.clone();
        out.attach(vec![self.clone()], move |g| {
            let one = Tensor::scalar_on(1.0, &g.device());
            vec![Some(g.mul(&one.sub(&o.mul(&o))))]
        })
    }

    pub fn sum(&self) -> Tensor {
        let shape = self.shape().to_vec();
        let out = self.raw_reshape(&[self.numel()]).raw_sum_axis(0);
        out.attach(vec![self.clone()], move |g| {
            vec![Some(g.mul(&Tensor::ones_on(&shape, &g.device())))]
        })
    }

    pub fn mean(&self) -> Tensor {
        let shape = self.shape().to_vec();
        let n = self.numel() as f32;
        let out = self.raw_reshape(&[self.numel()]).raw_sum_axis(0).div(&Tensor::scalar_on(n, &self.device()));
        out.attach(vec![self.clone()], move |g| {
            let dev = g.device();
            vec![Some(g.mul(&Tensor::ones_on(&shape, &dev)).div(&Tensor::scalar_on(n, &dev)))]
        })
    }

    /// 2-D matrix multiply.
    pub fn matmul(&self, o: &Tensor) -> Tensor {
        assert!(
            self.shape().len() == 2 && o.shape().len() == 2 && self.shape()[1] == o.shape()[0],
            "matmul needs [m,k] x [k,n], got {:?} x {:?}",
            self.shape(),
            o.shape()
        );
        let (m, n) = (self.shape()[0], o.shape()[1]);
        let out = run_op(
            &self.device(),
            op::MATMUL,
            [0; 4],
            &[Operand::contiguous(self), Operand::contiguous(o)],
            &[m, n],
        );
        let (a, b) = (self.detach(), o.detach());
        out.attach(vec![self.clone(), o.clone()], move |g| {
            vec![Some(g.matmul(&b.t())), Some(a.t().matmul(g))]
        })
    }

    /// 2-D transpose (materialized).
    pub fn t(&self) -> Tensor {
        self.raw_t().attach(vec![self.clone()], |g| vec![Some(g.t())])
    }

    pub fn reshape(&self, shape: &[usize]) -> Tensor {
        let old = self.shape().to_vec();
        self.raw_reshape(shape).attach(vec![self.clone()], move |g| vec![Some(g.reshape(&old))])
    }

    // ---- autograd ----------------------------------------------------------

    /// Reverse-mode backprop from a scalar tensor; accumulates into leaf `.grad`s.
    pub fn backward(&self) {
        assert_eq!(self.numel(), 1, "backward() needs a scalar tensor");
        assert!(self.requires_grad(), "tensor does not require grad");

        // Iterative post-order DFS -> topological order (inputs before consumers).
        let mut order: Vec<Tensor> = Vec::new();
        let mut visited: HashSet<u64> = HashSet::new();
        let mut stack: Vec<(Tensor, bool)> = vec![(self.clone(), false)];
        while let Some((t, expanded)) = stack.pop() {
            if expanded {
                order.push(t);
                continue;
            }
            if !visited.insert(t.0.id) {
                continue;
            }
            stack.push((t.clone(), true));
            if let Some(node) = &t.0.grad_fn {
                for inp in &node.inputs {
                    if inp.requires_grad() && !visited.contains(&inp.0.id) {
                        stack.push((inp.clone(), false));
                    }
                }
            }
        }

        let mut grads: HashMap<u64, Tensor> = HashMap::new();
        grads.insert(self.0.id, Tensor::ones_on(self.shape(), &self.device()));

        for t in order.into_iter().rev() {
            let Some(g) = grads.remove(&t.0.id) else { continue };
            match &t.0.grad_fn {
                Some(node) => {
                    let input_grads = (node.backward)(&g);
                    for (inp, gi) in node.inputs.iter().zip(input_grads) {
                        if let (true, Some(gi)) = (inp.requires_grad(), gi) {
                            let acc = match grads.remove(&inp.0.id) {
                                Some(prev) => prev.raw_binary(&gi, op::ADD),
                                None => gi,
                            };
                            grads.insert(inp.0.id, acc);
                        }
                    }
                }
                None => {
                    let mut slot = t.0.grad.lock().unwrap();
                    *slot = Some(match slot.take() {
                        Some(prev) => prev.raw_binary(&g, op::ADD),
                        None => g,
                    });
                }
            }
        }
    }
}

macro_rules! impl_op {
    ($trait:ident, $method:ident) => {
        impl $trait for &Tensor {
            type Output = Tensor;
            fn $method(self, rhs: &Tensor) -> Tensor {
                Tensor::$method(self, rhs)
            }
        }
    };
}
impl_op!(Add, add);
impl_op!(Sub, sub);
impl_op!(Mul, mul);
impl_op!(Div, div);

impl Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        Tensor::neg(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Once;

    fn init() {
        static ONCE: Once = Once::new();
        ONCE.call_once(|| {
            plugin::register_static(pytorches_plugin_cpu::pytorches_plugin_entry).unwrap();
        });
    }

    fn leaf(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        init();
        Tensor::new(data, shape).requires_grad_(true)
    }

    #[test]
    fn square_grad() {
        let x = leaf(vec![1.0, 2.0, 3.0], vec![3]);
        (&x * &x).sum().backward();
        assert_eq!(x.grad().unwrap().to_vec(), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn broadcast_grad_sums_over_batch() {
        let x = leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = leaf(vec![10.0, 20.0, 30.0], vec![3]);
        (&x + &b).sum().backward();
        assert_eq!(x.grad().unwrap().to_vec(), vec![1.0; 6]);
        assert_eq!(b.grad().unwrap().to_vec(), vec![2.0, 2.0, 2.0]);
    }

    #[test]
    fn broadcast_values() {
        let x = leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = leaf(vec![10.0, 20.0, 30.0], vec![3]);
        assert_eq!((&x + &b).to_vec(), vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn matmul_values_and_grad() {
        let a = leaf(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = leaf(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let c = a.matmul(&b);
        assert_eq!(c.to_vec(), vec![19.0, 22.0, 43.0, 50.0]);
        c.sum().backward();
        // dA = ones @ B^T, dB = A^T @ ones
        assert_eq!(a.grad().unwrap().to_vec(), vec![11.0, 15.0, 11.0, 15.0]);
        assert_eq!(b.grad().unwrap().to_vec(), vec![4.0, 4.0, 6.0, 6.0]);
    }

    #[test]
    fn transpose_values() {
        let x = leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        assert_eq!(x.t().to_vec(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn grads_accumulate_through_reuse() {
        let x = leaf(vec![3.0], vec![1]);
        // y = x*x + x  ->  dy/dx = 2x + 1 = 7
        (&(&x * &x) + &x).sum().backward();
        assert_eq!(x.grad().unwrap().to_vec(), vec![7.0]);
    }

    #[test]
    fn no_graph_without_requires_grad() {
        init();
        let x = Tensor::new(vec![1.0], vec![1]);
        assert!(!(&x * &x).requires_grad());
    }

    #[test]
    fn fill_and_copy_in_place() {
        init();
        let dev = Device::default_device();
        let t = Tensor::full_on(&[2, 2], 7.0, &dev);
        assert_eq!(t.to_vec(), vec![7.0; 4]);
        let alias = t.detach(); // shares storage
        t.copy_(&Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]));
        assert_eq!(alias.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(Tensor::zeros_on(&[3], &dev).to_vec(), vec![0.0; 3]);
    }

    #[test]
    fn randn_is_deterministic_and_normal() {
        init();
        let dev = Device::default_device();
        let a = Tensor::randn_on(&[100_000], 42, &dev).to_vec();
        assert_eq!(a, Tensor::randn_on(&[100_000], 42, &dev).to_vec());
        assert_ne!(a, Tensor::randn_on(&[100_000], 43, &dev).to_vec());
        let n = a.len() as f32;
        let mean = a.iter().sum::<f32>() / n;
        let var = a.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        assert!(mean.abs() < 0.02, "mean {mean}");
        assert!((var - 1.0).abs() < 0.03, "var {var}");
    }

    #[test]
    fn randn_golden_values() {
        // Pins the cross-device RNG recipe; GPU plugins must match these (within float tolerance).
        init();
        let v = Tensor::randn_on(&[4], 1234, &Device::default_device()).to_vec();
        for (got, want) in v.iter().zip([0.6574781f32, -0.08203645, -2.2065563, 0.70945626]) {
            assert!((got - want).abs() < 1e-5, "{v:?}");
        }
    }

    #[test]
    fn plan_with_cpu_only_picks_cpu() {
        init();
        let p = plan::plan(1 << 20);
        assert_eq!(p.chosen.to_string(), "cpu:0");
        assert!(!p.may_oom);
        assert!(p.candidates[0].gflops > 0.0);
    }

    #[test]
    fn errors_are_typed_not_aborts() {
        init();
        // shape mismatch -> Invalid, with the message preserved
        let a = Tensor::new(vec![1.0; 6], vec![2, 3]);
        let b = Tensor::new(vec![1.0; 6], vec![2, 3]);
        match try_run(|| a.matmul(&b)) {
            Err(Error::Invalid(m)) => assert!(m.contains("matmul"), "{m}"),
            other => panic!("expected Invalid, got {other:?}"),
        }
        // absurd allocation -> OutOfMemory (the cpu plugin reports STATUS_OUT_OF_MEMORY)
        let dev = Device::default_device();
        match try_run(|| Tensor::zeros_on(&[1 << 40, 1 << 20], &dev)) {
            Err(Error::OutOfMemory(m)) => assert!(m.contains("alloc"), "{m}"),
            other => panic!("expected OutOfMemory, got {other:?}"),
        }
        // and the library still works afterwards
        assert_eq!(try_run(|| Tensor::ones_on(&[2], &dev).to_vec()).unwrap(), vec![1.0, 1.0]);
    }

    #[test]
    fn borrowed_host_memory_is_zero_copy_and_released() {
        init();
        use std::sync::atomic::{AtomicBool, Ordering};
        let mut data = vec![1.0f32, 2.0, 3.0, 4.0];
        let released = Arc::new(AtomicBool::new(false));
        let flag = released.clone();
        let t = unsafe { Tensor::from_host_borrowed(data.as_mut_ptr(), vec![2, 2], move || flag.store(true, Ordering::SeqCst)) };
        assert_eq!(t.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
        data[0] = 9.0; // visible through the tensor: no copy was made
        assert_eq!(t.to_vec()[0], 9.0);
        assert_eq!((&t + &t).to_vec(), vec![18.0, 4.0, 6.0, 8.0]);
        assert!(!released.load(Ordering::SeqCst));
        drop(t);
        assert!(released.load(Ordering::SeqCst));
    }

    #[test]
    fn host_export_keeps_storage_alive() {
        init();
        let t = Tensor::new(vec![5.0, 6.0], vec![2]);
        let export = t.host_export().expect("cpu tensor exports");
        drop(t);
        let vals = unsafe { std::slice::from_raw_parts(export.ptr as *const f32, 2) };
        assert_eq!(vals, &[5.0, 6.0]);
    }

    #[test]
    fn device_parse() {
        init();
        assert_eq!(Device::parse("cpu").unwrap().to_string(), "cpu:0");
        assert!(Device::parse("nonexistent").is_err());
        assert!(Device::parse("cpu:5").is_err());
    }
}
