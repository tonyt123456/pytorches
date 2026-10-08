//! Native half of the Python package (`pytorches._native`); the public API lives in
//! `python/pytorches/`.

mod convert;
mod dlpack;

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyBufferError, PyMemoryError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use pytorches_core::graph::{GraphInfo, LayerInfo, Placement};
use pytorches_core::strategy::{self, MachineProfile, Proposal};
use pytorches_core::{Device, Error, Tensor, plan, plugin, try_run};

#[pyclass(name = "Tensor", module = "pytorches")]
#[derive(Clone)]
struct PyTensor(Tensor);

fn to_py_err(e: Error) -> PyErr {
    match e {
        Error::OutOfMemory(m) => PyMemoryError::new_err(m),
        Error::Invalid(m) => PyValueError::new_err(m),
        Error::Backend(m) => PyRuntimeError::new_err(m),
    }
}

/// Runs core work, mapping core errors to Python exceptions (`MemoryError`, `ValueError`,
/// `RuntimeError`) instead of letting a panic escape as `PanicException`.
fn guard<T>(f: impl FnOnce() -> T) -> PyResult<T> {
    try_run(f).map_err(to_py_err)
}

fn parse_device(s: &str) -> PyResult<Device> {
    Device::parse(s).map_err(PyValueError::new_err)
}

/// Accepts another Tensor or a Python number (a scalar tensor on `like`'s device).
fn coerce(o: &Bound<'_, PyAny>, like: &Tensor) -> PyResult<Tensor> {
    if let Ok(t) = o.extract::<PyTensor>() {
        Ok(t.0)
    } else if let Ok(f) = o.extract::<f64>() {
        Ok(Tensor::scalar_on(f as f32, &like.device()))
    } else {
        Err(PyTypeError::new_err("expected a pytorches.Tensor or a number"))
    }
}

#[pymethods]
impl PyTensor {
    /// `Tensor(flat_data, shape, requires_grad=False, device=None)`
    #[new]
    #[pyo3(signature = (data, shape, requires_grad=false, device=None))]
    fn new(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool, device: Option<&str>) -> PyResult<Self> {
        if data.len() != shape.iter().product::<usize>() {
            return Err(PyTypeError::new_err("data length does not match shape"));
        }
        let dev = match device {
            Some(d) => parse_device(d)?,
            None => guard(Device::default_device)?,
        };
        guard(|| PyTensor(Tensor::from_vec_on(data, shape, &dev).requires_grad_(requires_grad)))
    }

    #[getter]
    fn device(&self) -> String {
        self.0.device().to_string()
    }

    #[getter]
    fn nbytes(&self) -> usize {
        self.0.nbytes()
    }

    fn numel(&self) -> usize {
        self.0.numel()
    }

    /// Returns a leaf sharing this tensor's storage with the given `requires_grad`.
    fn requires_grad_(&self, flag: bool) -> PyTensor {
        PyTensor(self.0.requires_grad_(flag))
    }

    /// In-place overwrite (shapes must match); not tracked by autograd.
    fn copy_(&self, src: &PyTensor) -> PyResult<()> {
        guard(|| self.0.copy_(&src.0))
    }

    /// In-place `self += alpha * other` (shapes must match); not tracked by autograd.
    fn axpy_(&self, alpha: f32, other: &PyTensor) -> PyResult<()> {
        guard(|| self.0.axpy_(alpha, &other.0))
    }

    fn to(&self, device: &str) -> PyResult<PyTensor> {
        let dev = parse_device(device)?;
        guard(|| PyTensor(self.0.to(&dev)))
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.0.shape().to_vec()
    }

    #[getter]
    fn requires_grad(&self) -> bool {
        self.0.requires_grad()
    }

    #[getter]
    fn grad(&self) -> Option<PyTensor> {
        self.0.grad().map(PyTensor)
    }

    /// Flat row-major data.
    fn tolist(&self) -> PyResult<Vec<f32>> {
        guard(|| self.0.to_vec())
    }

    fn item(&self) -> PyResult<f32> {
        guard(|| self.0.item())
    }

    fn backward(&self) -> PyResult<()> {
        guard(|| self.0.backward())
    }

    fn zero_grad(&self) {
        self.0.zero_grad()
    }

    fn detach(&self) -> PyTensor {
        PyTensor(self.0.detach())
    }

    fn __add__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        let rhs = coerce(o, &self.0)?;
        guard(|| PyTensor(self.0.add(&rhs)))
    }
    fn __radd__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        let lhs = coerce(o, &self.0)?;
        guard(|| PyTensor(lhs.add(&self.0)))
    }
    fn __sub__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        let rhs = coerce(o, &self.0)?;
        guard(|| PyTensor(self.0.sub(&rhs)))
    }
    fn __rsub__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        let lhs = coerce(o, &self.0)?;
        guard(|| PyTensor(lhs.sub(&self.0)))
    }
    fn __mul__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        let rhs = coerce(o, &self.0)?;
        guard(|| PyTensor(self.0.mul(&rhs)))
    }
    fn __rmul__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        let lhs = coerce(o, &self.0)?;
        guard(|| PyTensor(lhs.mul(&self.0)))
    }
    fn __truediv__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        let rhs = coerce(o, &self.0)?;
        guard(|| PyTensor(self.0.div(&rhs)))
    }
    fn __rtruediv__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        let lhs = coerce(o, &self.0)?;
        guard(|| PyTensor(lhs.div(&self.0)))
    }
    fn __neg__(&self) -> PyResult<PyTensor> {
        guard(|| PyTensor(self.0.neg()))
    }
    fn __matmul__(&self, o: &PyTensor) -> PyResult<PyTensor> {
        guard(|| PyTensor(self.0.matmul(&o.0)))
    }

    fn matmul(&self, o: &PyTensor) -> PyResult<PyTensor> {
        guard(|| PyTensor(self.0.matmul(&o.0)))
    }
    fn exp(&self) -> PyResult<PyTensor> {
        guard(|| PyTensor(self.0.exp()))
    }
    fn log(&self) -> PyResult<PyTensor> {
        guard(|| PyTensor(self.0.log()))
    }
    fn relu(&self) -> PyResult<PyTensor> {
        guard(|| PyTensor(self.0.relu()))
    }
    fn tanh(&self) -> PyResult<PyTensor> {
        guard(|| PyTensor(self.0.tanh()))
    }
    fn sum(&self) -> PyResult<PyTensor> {
        guard(|| PyTensor(self.0.sum()))
    }
    fn mean(&self) -> PyResult<PyTensor> {
        guard(|| PyTensor(self.0.mean()))
    }
    #[pyo3(name = "t")]
    fn transpose(&self) -> PyResult<PyTensor> {
        guard(|| PyTensor(self.0.t()))
    }
    fn reshape(&self, shape: Vec<usize>) -> PyResult<PyTensor> {
        guard(|| PyTensor(self.0.reshape(&shape)))
    }

    /// DLPack export (zero-copy for CPU tensors). Only `copy` and `dl_device` are acted on:
    /// the CPU needs no stream synchronization.
    #[pyo3(signature = (stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &self,
        py: Python<'py>,
        stream: Option<Bound<'py, PyAny>>,
        max_version: Option<Bound<'py, PyAny>>,
        dl_device: Option<(i32, i32)>,
        copy: Option<bool>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let _ = (stream, max_version);
        if let Some(d) = dl_device {
            if d != (1, 0) {
                return Err(PyBufferError::new_err(format!(
                    "can only export to the CPU (DLPack device (1, 0)), asked for {d:?}"
                )));
            }
        }
        let t = if copy == Some(true) {
            guard(|| {
                let dev = Device::parse("cpu").expect("cpu plugin");
                Tensor::from_vec_on(self.0.to_vec(), self.0.shape().to_vec(), &dev)
            })?
        } else {
            self.0.clone()
        };
        dlpack::export(py, &t)
    }

    fn __dlpack_device__(&self) -> PyResult<(i32, i32)> {
        let on_cpu = guard(|| self.0.host_export().is_some())?;
        if on_cpu {
            Ok((1, 0))
        } else {
            Err(PyBufferError::new_err(format!(
                "tensor is on {}; DLPack export is zero-copy for CPU tensors only. Use t.to('cpu:0') first",
                self.0.device()
            )))
        }
    }

    fn __repr__(&self) -> PyResult<String> {
        guard(|| format!("{:?}", self.0))
    }
}

/// Loads every `pytorches_plugin_*` library in `path`; returns the loaded plugin names.
#[pyfunction]
fn load_plugins(path: &str) -> PyResult<Vec<String>> {
    let mut loaded = Vec::new();
    let mut errors = Vec::new();
    for (_, r) in plugin::load_plugin_dir(std::path::Path::new(path)) {
        match r {
            Ok(name) => loaded.push(name),
            Err(e) => errors.push(e),
        }
    }
    if loaded.is_empty() && !errors.is_empty() {
        return Err(PyRuntimeError::new_err(errors.join("; ")));
    }
    Ok(loaded)
}

/// Names of all available devices, e.g. `["cpu:0", "cuda:0"]`.
#[pyfunction]
fn devices() -> Vec<String> {
    Device::all().iter().map(|d| d.to_string()).collect()
}

#[pyfunction]
fn device_info<'py>(py: Python<'py>, device: &str) -> PyResult<Bound<'py, PyDict>> {
    let dev = parse_device(device)?;
    let info = guard(|| dev.info())?;
    let d = PyDict::new(py);
    d.set_item("name", info.name)?;
    d.set_item("kind", info.kind)?;
    d.set_item("total_memory", info.total_memory)?;
    d.set_item("free_memory", info.free_memory)?;
    d.set_item("shared_host_memory", info.shared_host_memory)?;
    Ok(d)
}

fn dev_or_default(device: Option<&str>) -> PyResult<Device> {
    match device {
        Some(d) => parse_device(d),
        None => guard(Device::default_device),
    }
}

/// Tensor of N(0,1) samples generated on the device.
#[pyfunction]
#[pyo3(signature = (shape, device=None, seed=0, requires_grad=false))]
fn randn(shape: Vec<usize>, device: Option<&str>, seed: u64, requires_grad: bool) -> PyResult<PyTensor> {
    let dev = dev_or_default(device)?;
    guard(|| PyTensor(Tensor::randn_on(&shape, seed, &dev).requires_grad_(requires_grad)))
}

/// Constant-filled tensor created directly on the device.
#[pyfunction]
#[pyo3(signature = (shape, value, device=None, requires_grad=false))]
fn full(shape: Vec<usize>, value: f32, device: Option<&str>, requires_grad: bool) -> PyResult<PyTensor> {
    let dev = dev_or_default(device)?;
    guard(|| PyTensor(Tensor::full_on(&shape, value, &dev).requires_grad_(requires_grad)))
}

#[pyfunction]
fn synchronize(device: &str) -> PyResult<()> {
    let dev = parse_device(device)?;
    guard(|| dev.synchronize())
}

/// Measured matmul throughput of a device in GFLOP/s.
#[pyfunction]
fn calibrate(device: &str) -> PyResult<f64> {
    let dev = parse_device(device)?;
    guard(|| plan::calibrate(&dev))
}

/// What plugin discovery tried at import: a list of `(file, loaded_plugin_name_or_None, error_or_None)`.
#[pyfunction]
fn plugin_report() -> Vec<(String, Option<String>, Option<String>)> {
    plugin::discovery_report()
        .into_iter()
        .map(|(path, r)| {
            let file = path.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
            match r {
                Ok(name) => (file, Some(name), None),
                Err(e) => (file, None, Some(e)),
            }
        })
        .collect()
}

/// Picks a device for a workload needing `required_bytes`; returns a dict describing the decision.
#[pyfunction]
fn plan_placement<'py>(py: Python<'py>, required_bytes: u64) -> PyResult<Bound<'py, PyDict>> {
    let p = guard(|| plan::plan(required_bytes))?;
    let out = PyDict::new(py);
    out.set_item("required_bytes", p.required_bytes)?;
    out.set_item("chosen", p.chosen.to_string())?;
    out.set_item("reason", p.reason)?;
    out.set_item("may_oom", p.may_oom)?;
    out.set_item("warnings", p.warnings)?;
    let mut cands = Vec::new();
    for c in p.candidates {
        let d = PyDict::new(py);
        d.set_item("device", c.device.to_string())?;
        d.set_item("name", c.name)?;
        d.set_item("kind", c.kind)?;
        d.set_item("total_memory", c.total_memory)?;
        d.set_item("free_memory", c.free_memory)?;
        d.set_item("shared_host_memory", c.shared_host_memory)?;
        d.set_item("gflops", c.gflops)?;
        d.set_item("fits", c.fits)?;
        cands.push(d);
    }
    out.set_item("candidates", cands)?;
    Ok(out)
}

/// A model described as a chain of layers, for placement planning. Holds no tensors.
///
/// `layers` is a list of `(name, param_bytes, out_bytes, flops)`.
#[pyclass(name = "GraphInfo", module = "pytorches")]
#[derive(Clone)]
struct PyGraphInfo(GraphInfo);

#[pymethods]
impl PyGraphInfo {
    #[new]
    fn new(input_bytes: u64, layers: Vec<(String, u64, u64, u64)>) -> Self {
        let layers = layers
            .into_iter()
            .map(|(name, param_bytes, out_bytes, flops)| LayerInfo { name, param_bytes, out_bytes, flops })
            .collect();
        PyGraphInfo(GraphInfo { input_bytes, layers })
    }

    #[getter]
    fn input_bytes(&self) -> u64 {
        self.0.input_bytes
    }

    /// `[(name, param_bytes, out_bytes, flops, training_bytes), ...]`
    #[getter]
    fn layers(&self) -> Vec<(String, u64, u64, u64, u64)> {
        self.0
            .layers
            .iter()
            .map(|l| (l.name.clone(), l.param_bytes, l.out_bytes, l.flops, l.training_bytes()))
            .collect()
    }

    #[getter]
    fn total_param_bytes(&self) -> u64 {
        self.0.total_param_bytes()
    }

    #[getter]
    fn total_training_bytes(&self) -> u64 {
        self.0.total_training_bytes()
    }

    #[getter]
    fn total_flops(&self) -> u64 {
        self.0.total_flops()
    }

    /// Prices a placement (one device string per layer): returns a dict with `per_device`
    /// (`[(device, bytes)]`), `transfers` (`[(after_layer, from, to, bytes)]`), `transfer_bytes`
    /// and `host_pool_bytes` (the load on system RAM from the CPU and every shared-memory device).
    fn price<'py>(&self, py: Python<'py>, devices: Vec<String>) -> PyResult<Bound<'py, PyDict>> {
        let devs = devices.iter().map(|d| parse_device(d)).collect::<PyResult<Vec<_>>>()?;
        let report = guard(|| Placement::new(devs).report(&self.0))?.map_err(PyValueError::new_err)?;
        let out = PyDict::new(py);
        out.set_item(
            "per_device",
            report.per_device.iter().map(|(d, b)| (d.to_string(), *b)).collect::<Vec<_>>(),
        )?;
        out.set_item(
            "transfers",
            report
                .transfers
                .iter()
                .map(|t| (t.after_layer, t.from.to_string(), t.to.to_string(), t.bytes))
                .collect::<Vec<_>>(),
        )?;
        out.set_item("transfer_bytes", report.transfer_bytes)?;
        out.set_item("host_pool_bytes", report.host_pool_bytes)?;
        Ok(out)
    }
}

fn proposal_dict<'py>(
    py: Python<'py>,
    p: &Proposal,
    machine: &MachineProfile,
) -> PyResult<Bound<'py, PyDict>> {
    let id = |i: usize| machine.devices[i].id.clone();
    let d = PyDict::new(py);
    d.set_item("strategy", &p.strategy)?;
    d.set_item("devices", p.assignment.iter().map(|&i| id(i)).collect::<Vec<_>>())?;
    d.set_item("est_step_secs", p.est_step_secs)?;
    d.set_item("compute_secs", p.compute_secs)?;
    d.set_item("transfer_secs", p.transfer_secs)?;
    d.set_item("reason", &p.reason)?;
    d.set_item("per_device", p.pricing.per_device.iter().map(|&(i, b)| (id(i), b)).collect::<Vec<_>>())?;
    d.set_item("host_pool_bytes", p.pricing.host_pool_bytes)?;
    // (after_layer, from, to, one_way_bytes, seconds for the activation and its gradient)
    let transfers: Vec<_> = p
        .pricing
        .transfers
        .iter()
        .map(|t| {
            let one_way = t.bytes / 2;
            let secs = machine.link(t.from, t.to).secs(one_way) + machine.link(t.to, t.from).secs(one_way);
            (t.after_layer, id(t.from), id(t.to), one_way, secs)
        })
        .collect();
    d.set_item("transfers", transfers)?;
    Ok(d)
}

/// Chooses a placement for `graph` by asking the placement strategies (all of them, or only the
/// one named `strategy`). Returns a dict: `chosen` and `considered` proposals (fastest first),
/// `declined` (`[(strategy, reason)]`) and `machine` (the measured device profiles).
#[pyfunction]
#[pyo3(signature = (graph, strategy=None))]
fn plan_model<'py>(py: Python<'py>, graph: &PyGraphInfo, strategy: Option<&str>) -> PyResult<Bound<'py, PyDict>> {
    let machine = guard(MachineProfile::measure)?;
    let plan = strategy::choose(&strategy::default_strategies(), &graph.0, &machine, strategy)
        .map_err(|e| match e {
            strategy::ChooseError::Invalid(m) => PyValueError::new_err(m),
            strategy::ChooseError::NoPlacement(m) => PyMemoryError::new_err(m),
        })?;
    let out = PyDict::new(py);
    out.set_item("chosen", proposal_dict(py, &plan.chosen, &machine)?)?;
    let considered = plan.considered.iter().map(|p| proposal_dict(py, p, &machine)).collect::<PyResult<Vec<_>>>()?;
    out.set_item("considered", considered)?;
    out.set_item("declined", plan.declined)?;
    let devices = machine
        .devices
        .iter()
        .map(|d| {
            let e = PyDict::new(py);
            e.set_item("device", &d.id)?;
            e.set_item("name", &d.name)?;
            e.set_item("gflops", d.gflops)?;
            e.set_item("gbps", d.gbps)?;
            e.set_item("free_bytes", d.free_bytes)?;
            e.set_item("shared_host_memory", d.shared_host)?;
            Ok(e)
        })
        .collect::<PyResult<Vec<_>>>()?;
    out.set_item("machine", devices)?;
    Ok(out)
}

/// Names of the built-in placement strategies.
#[pyfunction]
fn strategies() -> Vec<String> {
    strategy::default_strategies().iter().map(|s| s.name().to_string()).collect()
}

/// Imports any object implementing `__dlpack__` (torch, numpy, jax, ...) as a CPU tensor.
/// Zero-copy for contiguous float32 CPU producers; other dtypes/layouts are converted with one copy.
#[pyfunction]
fn from_dlpack(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
    Ok(PyTensor(dlpack::import(py, obj)?))
}

/// Builds a float32 tensor from raw little-endian bytes of the given safetensors dtype
/// (`"F32"`, `"F16"`, `"BF16"`, `"I64"`, ...), widening to float32.
#[pyfunction]
#[pyo3(signature = (buffer, dtype, shape, device=None))]
fn tensor_from_buffer(buffer: PyBuffer<u8>, dtype: &str, shape: Vec<usize>, device: Option<&str>) -> PyResult<PyTensor> {
    let elem = convert::Elem::from_safetensors(dtype)
        .ok_or_else(|| PyValueError::new_err(format!("unsupported safetensors dtype '{dtype}'")))?;
    if !buffer.is_c_contiguous() {
        return Err(PyBufferError::new_err("buffer must be C-contiguous"));
    }
    let n: usize = shape.iter().product();
    if buffer.len_bytes() != n * elem.size() {
        return Err(PyValueError::new_err(format!(
            "buffer has {} bytes, shape {shape:?} of {dtype} needs {}",
            buffer.len_bytes(),
            n * elem.size()
        )));
    }
    let dev = dev_or_default(device)?;
    let base = buffer.buf_ptr() as *const u8;
    let mut data = vec![0.0f32; n];
    unsafe {
        if elem == convert::Elem::F32 {
            std::ptr::copy_nonoverlapping(base, data.as_mut_ptr() as *mut u8, n * 4);
        } else {
            for (i, v) in data.iter_mut().enumerate() {
                *v = elem.read(base.add(i * elem.size()));
            }
        }
    }
    guard(|| PyTensor(Tensor::from_vec_on(data, shape, &dev)))
}

/// The tensor's data as little-endian float32 bytes.
#[pyfunction]
fn tensor_to_bytes<'py>(py: Python<'py>, t: &PyTensor) -> PyResult<Bound<'py, PyBytes>> {
    let data = guard(|| t.0.to_vec())?;
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    Ok(PyBytes::new(py, &bytes))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Auto-discover plugins ($PYTORCHES_PLUGIN_DIR, set by the package; else ./plugins/bin).
    plugin::discover();
    m.add_class::<PyTensor>()?;
    m.add_class::<PyGraphInfo>()?;
    m.add_function(wrap_pyfunction!(randn, m)?)?;
    m.add_function(wrap_pyfunction!(full, m)?)?;
    m.add_function(wrap_pyfunction!(synchronize, m)?)?;
    m.add_function(wrap_pyfunction!(from_dlpack, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_from_buffer, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_to_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(calibrate, m)?)?;
    m.add_function(wrap_pyfunction!(plugin_report, m)?)?;
    m.add_function(wrap_pyfunction!(plan_placement, m)?)?;
    m.add_function(wrap_pyfunction!(plan_model, m)?)?;
    m.add_function(wrap_pyfunction!(strategies, m)?)?;
    m.add_function(wrap_pyfunction!(load_plugins, m)?)?;
    m.add_function(wrap_pyfunction!(devices, m)?)?;
    m.add_function(wrap_pyfunction!(device_info, m)?)?;
    Ok(())
}
