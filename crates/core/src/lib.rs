//! PyTorches core: tensors and reverse-mode autograd.
//!
//! The core contains no device code. Every tensor lives in a buffer owned by a plugin
//! (see [`plugin`]) and every op is dispatched to that plugin through the C ABI in
//! `pytorches-plugin-abi`.
//!
//! Current scope: `f32`, contiguous tensors. Broadcasting and transposes are expressed
//! as strided operands at dispatch time rather than as stored views.

pub mod plugin;

use plugin::Plugin;
use pytorches_plugin_abi::{self as abi, OpAttrs, TensorDesc, op};
use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

fn next_id() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    NEXT.fetch_add(1, Ordering::Relaxed)
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
}

// The pointer is opaque and only ever used via thread-safe plugin entry points.
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    fn alloc(device: &Device, numel: usize) -> Buffer {
        let mut ptr = std::ptr::null_mut();
        let bytes = (numel * 4).max(4);
        let status = unsafe { (device.plugin.vt.alloc)(device.index, bytes, &mut ptr) };
        assert_eq!(status, abi::STATUS_OK, "alloc of {bytes} bytes on {device} failed: {}", device.plugin.last_error());
        Buffer { device: device.clone(), ptr }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe { (self.device.plugin.vt.free)(self.device.index, self.ptr) };
    }
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
    let vt = device.plugin.vt;
    assert!(
        unsafe { (vt.supports_op)(code) } != 0,
        "plugin '{}' does not implement op {code}",
        device.plugin.name
    );
    for o in ins {
        assert!(&o.t.device() == device, "tensors on different devices: {} and {device}", o.t.device());
    }
    let out_buf = Arc::new(Buffer::alloc(device, out_shape.iter().product()));

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
    assert_eq!(status, abi::STATUS_OK, "op {code} failed on {device}: {}", device.plugin.last_error());
    Tensor::build(out_buf, out_shape.to_vec(), false, None)
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
        assert_eq!(status, abi::STATUS_OK, "host->device copy failed: {}", device.plugin.last_error());
        Self::build(Arc::new(buf), shape, false, None)
    }

    /// On the default device.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self::from_vec_on(data, shape, &Device::default_device())
    }

    pub fn scalar_on(v: f32, device: &Device) -> Self {
        Self::from_vec_on(vec![v], vec![], device)
    }

    pub fn ones_on(shape: &[usize], device: &Device) -> Self {
        Self::from_vec_on(vec![1.0; shape.iter().product()], shape.to_vec(), device)
    }

    pub fn zeros_on(shape: &[usize], device: &Device) -> Self {
        Self::from_vec_on(vec![0.0; shape.iter().product()], shape.to_vec(), device)
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

    /// Copies the data to the host.
    pub fn to_vec(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; self.numel()];
        let dev = &self.0.data.device;
        let status = unsafe {
            (dev.plugin.vt.copy_to_host)(dev.index, out.as_mut_ptr() as *mut c_void, self.0.data.ptr, out.len() * 4)
        };
        assert_eq!(status, abi::STATUS_OK, "device->host copy failed: {}", dev.plugin.last_error());
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
    fn device_parse() {
        init();
        assert_eq!(Device::parse("cpu").unwrap().to_string(), "cpu:0");
        assert!(Device::parse("nonexistent").is_err());
        assert!(Device::parse("cpu:5").is_err());
    }
}
