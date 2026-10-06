//! Python bindings: `import pytorches`.

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pytorches_core::{Device, Tensor, plugin};

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

#[pymodule]
fn pytorches(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Auto-discover plugins ($PYTORCHES_PLUGIN_DIR, else ./plugins/bin).
    plugin::discover();
    m.add_class::<PyTensor>()?;
    m.add_function(wrap_pyfunction!(load_plugins, m)?)?;
    m.add_function(wrap_pyfunction!(devices, m)?)?;
    m.add_function(wrap_pyfunction!(device_info, m)?)?;
    Ok(())
}
