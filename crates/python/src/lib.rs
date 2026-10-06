//! Native half of the Python package (`pytorches._native`); the public API lives in
//! `python/pytorches/`.

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pytorches_core::{Device, Tensor, plan, plugin};

#[pyclass(name = "Tensor", module = "pytorches")]
#[derive(Clone)]
struct PyTensor(Tensor);

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
            None => Device::default_device(),
        };
        Ok(PyTensor(Tensor::from_vec_on(data, shape, &dev).requires_grad_(requires_grad)))
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
    fn copy_(&self, src: &PyTensor) {
        self.0.copy_(&src.0)
    }

    fn to(&self, device: &str) -> PyResult<PyTensor> {
        Ok(PyTensor(self.0.to(&parse_device(device)?)))
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
    fn tolist(&self) -> Vec<f32> {
        self.0.to_vec()
    }

    fn item(&self) -> f32 {
        self.0.item()
    }

    fn backward(&self) {
        self.0.backward()
    }

    fn zero_grad(&self) {
        self.0.zero_grad()
    }

    fn detach(&self) -> PyTensor {
        PyTensor(self.0.detach())
    }

    fn __add__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        Ok(PyTensor(self.0.add(&coerce(o, &self.0)?)))
    }
    fn __radd__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        Ok(PyTensor(coerce(o, &self.0)?.add(&self.0)))
    }
    fn __sub__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        Ok(PyTensor(self.0.sub(&coerce(o, &self.0)?)))
    }
    fn __rsub__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        Ok(PyTensor(coerce(o, &self.0)?.sub(&self.0)))
    }
    fn __mul__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        Ok(PyTensor(self.0.mul(&coerce(o, &self.0)?)))
    }
    fn __rmul__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        Ok(PyTensor(coerce(o, &self.0)?.mul(&self.0)))
    }
    fn __truediv__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        Ok(PyTensor(self.0.div(&coerce(o, &self.0)?)))
    }
    fn __rtruediv__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        Ok(PyTensor(coerce(o, &self.0)?.div(&self.0)))
    }
    fn __neg__(&self) -> PyTensor {
        PyTensor(self.0.neg())
    }
    fn __matmul__(&self, o: &PyTensor) -> PyTensor {
        PyTensor(self.0.matmul(&o.0))
    }

    fn matmul(&self, o: &PyTensor) -> PyTensor {
        PyTensor(self.0.matmul(&o.0))
    }
    fn exp(&self) -> PyTensor {
        PyTensor(self.0.exp())
    }
    fn log(&self) -> PyTensor {
        PyTensor(self.0.log())
    }
    fn relu(&self) -> PyTensor {
        PyTensor(self.0.relu())
    }
    fn tanh(&self) -> PyTensor {
        PyTensor(self.0.tanh())
    }
    fn sum(&self) -> PyTensor {
        PyTensor(self.0.sum())
    }
    fn mean(&self) -> PyTensor {
        PyTensor(self.0.mean())
    }
    #[pyo3(name = "t")]
    fn transpose(&self) -> PyTensor {
        PyTensor(self.0.t())
    }
    fn reshape(&self, shape: Vec<usize>) -> PyTensor {
        PyTensor(self.0.reshape(&shape))
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
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
    let info = parse_device(device)?.info();
    let d = PyDict::new(py);
    d.set_item("name", info.name)?;
    d.set_item("kind", info.kind)?;
    d.set_item("total_memory", info.total_memory)?;
    d.set_item("free_memory", info.free_memory)?;
    Ok(d)
}

fn dev_or_default(device: Option<&str>) -> PyResult<Device> {
    match device {
        Some(d) => parse_device(d),
        None => Ok(Device::default_device()),
    }
}

/// Tensor of N(0,1) samples generated on the device.
#[pyfunction]
#[pyo3(signature = (shape, device=None, seed=0, requires_grad=false))]
fn randn(shape: Vec<usize>, device: Option<&str>, seed: u64, requires_grad: bool) -> PyResult<PyTensor> {
    let t = Tensor::randn_on(&shape, seed, &dev_or_default(device)?);
    Ok(PyTensor(t.requires_grad_(requires_grad)))
}

/// Constant-filled tensor created directly on the device.
#[pyfunction]
#[pyo3(signature = (shape, value, device=None, requires_grad=false))]
fn full(shape: Vec<usize>, value: f32, device: Option<&str>, requires_grad: bool) -> PyResult<PyTensor> {
    let t = Tensor::full_on(&shape, value, &dev_or_default(device)?);
    Ok(PyTensor(t.requires_grad_(requires_grad)))
}

#[pyfunction]
fn synchronize(device: &str) -> PyResult<()> {
    parse_device(device)?.synchronize();
    Ok(())
}

/// Measured matmul throughput of a device in GFLOP/s.
#[pyfunction]
fn calibrate(device: &str) -> PyResult<f64> {
    Ok(plan::calibrate(&parse_device(device)?))
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
    let p = plan::plan(required_bytes);
    let out = PyDict::new(py);
    out.set_item("required_bytes", p.required_bytes)?;
    out.set_item("chosen", p.chosen.to_string())?;
    out.set_item("reason", p.reason)?;
    out.set_item("may_oom", p.may_oom)?;
    let mut cands = Vec::new();
    for c in p.candidates {
        let d = PyDict::new(py);
        d.set_item("device", c.device.to_string())?;
        d.set_item("name", c.name)?;
        d.set_item("kind", c.kind)?;
        d.set_item("total_memory", c.total_memory)?;
        d.set_item("free_memory", c.free_memory)?;
        d.set_item("gflops", c.gflops)?;
        d.set_item("fits", c.fits)?;
        cands.push(d);
    }
    out.set_item("candidates", cands)?;
    Ok(out)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Auto-discover plugins ($PYTORCHES_PLUGIN_DIR, set by the package; else ./plugins/bin).
    plugin::discover();
    m.add_class::<PyTensor>()?;
    m.add_function(wrap_pyfunction!(randn, m)?)?;
    m.add_function(wrap_pyfunction!(full, m)?)?;
    m.add_function(wrap_pyfunction!(synchronize, m)?)?;
    m.add_function(wrap_pyfunction!(calibrate, m)?)?;
    m.add_function(wrap_pyfunction!(plugin_report, m)?)?;
    m.add_function(wrap_pyfunction!(plan_placement, m)?)?;
    m.add_function(wrap_pyfunction!(load_plugins, m)?)?;
    m.add_function(wrap_pyfunction!(devices, m)?)?;
    m.add_function(wrap_pyfunction!(device_info, m)?)?;
    Ok(())
}
