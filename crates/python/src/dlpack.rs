//! DLPack interchange (https://dmlc.github.io/dlpack/): zero-copy for CPU tensors in both
//! directions; other devices go through a host copy.
//!
//! Uses the legacy (unversioned) capsule protocol, `"dltensor"` / `"used_dltensor"`, which every
//! major framework accepts.

use crate::convert::{Elem, contiguous_strides, gather_f32, is_contiguous};
use pyo3::exceptions::{PyBufferError, PyTypeError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pytorches_core::{Device, Tensor};
use std::ffi::c_void;

const DL_CPU: i32 = 1;

#[repr(C)]
#[derive(Clone, Copy)]
struct DLDevice {
    device_type: i32,
    device_id: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DLDataType {
    code: u8,
    bits: u8,
    lanes: u16,
}

#[repr(C)]
struct DLTensor {
    data: *mut c_void,
    device: DLDevice,
    ndim: i32,
    dtype: DLDataType,
    shape: *mut i64,
    strides: *mut i64,
    byte_offset: u64,
}

#[repr(C)]
struct DLManagedTensor {
    dl_tensor: DLTensor,
    manager_ctx: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

const NAME_LIVE: &std::ffi::CStr = c"dltensor";
const NAME_USED: &std::ffi::CStr = c"used_dltensor";

// ---- export ---------------------------------------------------------------------

/// Owned by the managed tensor; freed by its deleter.
struct ExportCtx {
    _keepalive: Box<dyn std::any::Any + Send + Sync>,
    _shape: Vec<i64>,
    _strides: Vec<i64>,
}

unsafe extern "C" fn export_deleter(m: *mut DLManagedTensor) {
    unsafe {
        let managed = Box::from_raw(m);
        drop(Box::from_raw(managed.manager_ctx as *mut ExportCtx));
    }
}

/// Capsule destructor: if nobody consumed the capsule, release the tensor ourselves.
unsafe extern "C" fn capsule_destructor(capsule: *mut ffi::PyObject) {
    unsafe {
        if ffi::PyCapsule_IsValid(capsule, NAME_LIVE.as_ptr()) == 1 {
            let m = ffi::PyCapsule_GetPointer(capsule, NAME_LIVE.as_ptr()) as *mut DLManagedTensor;
            if !m.is_null() {
                if let Some(deleter) = (*m).deleter {
                    deleter(m);
                }
            }
        }
    }
}

/// Exports a CPU tensor as a `"dltensor"` capsule sharing its memory.
pub fn export<'py>(py: Python<'py>, t: &Tensor) -> PyResult<Bound<'py, PyAny>> {
    let host = t.host_export().ok_or_else(|| {
        PyBufferError::new_err(format!(
            "tensor is on {}; DLPack export is zero-copy for CPU tensors only. Use t.to('cpu:0') first",
            t.device()
        ))
    })?;
    let mut shape: Vec<i64> = t.shape().iter().map(|&d| d as i64).collect();
    let mut strides = contiguous_strides(t.shape());
    let dl_tensor = DLTensor {
        data: host.ptr,
        device: DLDevice { device_type: DL_CPU, device_id: 0 },
        ndim: shape.len() as i32,
        dtype: DLDataType { code: 2, bits: 32, lanes: 1 },
        shape: shape.as_mut_ptr(),
        strides: strides.as_mut_ptr(),
        byte_offset: 0,
    };
    // Moving the Vecs into the context doesn't move their heap buffers, so the pointers stay valid.
    let ctx = Box::new(ExportCtx { _keepalive: host.keepalive, _shape: shape, _strides: strides });
    let managed = Box::new(DLManagedTensor {
        dl_tensor,
        manager_ctx: Box::into_raw(ctx) as *mut c_void,
        deleter: Some(export_deleter),
    });
    let raw = Box::into_raw(managed);
    unsafe {
        let capsule = ffi::PyCapsule_New(raw as *mut c_void, NAME_LIVE.as_ptr(), Some(capsule_destructor));
        if capsule.is_null() {
            export_deleter(raw);
            return Err(PyErr::fetch(py));
        }
        Ok(Bound::from_owned_ptr(py, capsule))
    }
}

// ---- import ---------------------------------------------------------------------

struct SendPtr(*mut DLManagedTensor);
unsafe impl Send for SendPtr {}

fn call_deleter(m: *mut DLManagedTensor) {
    unsafe {
        if let Some(deleter) = (*m).deleter {
            deleter(m);
        }
    }
}

/// Imports any object implementing `__dlpack__` as a CPU tensor.
///
/// A contiguous, aligned `f32` CPU producer is shared with no copy (the tensor keeps the
/// producer's memory alive). Other dtypes and strided layouts are converted to contiguous `f32`
/// with one copy. Producers on other devices are asked to copy to the CPU.
pub fn import(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Tensor> {
    if !obj.hasattr("__dlpack__")? {
        return Err(PyTypeError::new_err("object does not implement __dlpack__"));
    }
    let (device_type, device_id): (i32, i32) = if obj.hasattr("__dlpack_device__")? {
        obj.call_method0("__dlpack_device__")?.extract()?
    } else {
        (DL_CPU, 0)
    };

    let capsule = if device_type == DL_CPU {
        obj.call_method0("__dlpack__")?
    } else {
        let kwargs = PyDict::new(py);
        kwargs.set_item("dl_device", (DL_CPU, 0))?;
        kwargs.set_item("copy", true)?;
        obj.call_method("__dlpack__", (), Some(&kwargs)).map_err(|e| {
            PyBufferError::new_err(format!(
                "producer is on DLPack device type {device_type}:{device_id} and could not copy to CPU ({e}); \
                 move it to the CPU first (e.g. tensor.cpu())"
            ))
        })?
    };

    unsafe {
        let cap = capsule.as_ptr();
        if ffi::PyCapsule_IsValid(cap, NAME_LIVE.as_ptr()) != 1 {
            return Err(PyTypeError::new_err(
                "expected an unconsumed 'dltensor' capsule (versioned DLPack capsules are not supported yet)",
            ));
        }
        let m = ffi::PyCapsule_GetPointer(cap, NAME_LIVE.as_ptr()) as *mut DLManagedTensor;
        // Take ownership: mark the capsule consumed so its destructor leaves the tensor alone.
        ffi::PyCapsule_SetName(cap, NAME_USED.as_ptr());

        let dl = &(*m).dl_tensor;
        let result = (|| -> PyResult<Tensor> {
            if dl.device.device_type != DL_CPU {
                return Err(PyBufferError::new_err(format!(
                    "DLPack device type {} is not CPU-accessible",
                    dl.device.device_type
                )));
            }
            let elem = Elem::from_dlpack(dl.dtype.code, dl.dtype.bits, dl.dtype.lanes).ok_or_else(|| {
                PyTypeError::new_err(format!(
                    "unsupported DLPack dtype (code {}, bits {}, lanes {})",
                    dl.dtype.code, dl.dtype.bits, dl.dtype.lanes
                ))
            })?;
            let ndim = dl.ndim as usize;
            let shape: Vec<usize> = std::slice::from_raw_parts(dl.shape, ndim)
                .iter()
                .map(|&d| usize::try_from(d))
                .collect::<Result<_, _>>()
                .map_err(|_| PyBufferError::new_err("negative dimension in DLPack shape"))?;
            let strides: Vec<i64> = if dl.strides.is_null() {
                contiguous_strides(&shape)
            } else {
                std::slice::from_raw_parts(dl.strides, ndim).to_vec()
            };
            let base = (dl.data as *const u8).add(dl.byte_offset as usize);
            Ok(if elem == Elem::F32 && is_contiguous(&shape, &strides) {
                // Zero copy: the tensor keeps the producer's memory alive until it is dropped.
                let guard = SendPtr(m);
                Tensor::from_host_borrowed(base as *mut f32, shape, move || {
                    let g = guard;
                    call_deleter(g.0)
                })
            } else {
                let data = gather_f32(base, elem, &shape, &strides);
                call_deleter(m);
                Tensor::from_vec_on(data, shape, &Device::parse("cpu").expect("cpu plugin"))
            })
        })();
        if result.is_err() {
            call_deleter(m);
        }
        result
    }
}
