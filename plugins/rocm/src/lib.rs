//! AMD ROCm/HIP plugin: **placeholder**.
//!
//! This crate exports a valid plugin entry point so the packaging, loader and `pt.doctor()`
//! paths can be exercised, but it has no backend yet: it reports zero devices, so the core
//! never selects it, and it implements no ops.
//!
//! Plan for the real backend:
//! * Load `libamdhip64.so` (Linux) / `amdhip64.dll` (Windows) at runtime with `libloading`, as the
//!   CUDA plugin does with the driver API, so no ROCm SDK is needed to build or install.
//! * Kernels as HIP code objects (gfx942/950, RDNA3/4) compiled ahead of time and committed, or
//!   compiled at first use through hipRTC; hipBLAS (loaded dynamically) for matmul.
//! * Targets the shared kernel IR/codegen item in TODO.md (Phase 8) rather than hand-written kernels.
//!
//! To make it real: implement `device_count`/`device_info` against the runtime, then
//! `alloc`/`free`/`copy_*`, `synchronize`, and advertise ops through `supports_op` as they
//! are implemented. The ABI conformance suite (see `plugins/cuda/tests`) is the exit test.
use core::ffi::{c_char, c_void};
use pytorches_plugin_abi::*;

/// Kind reported by `device_info` once devices exist.
#[allow(dead_code)]
const KIND: u32 = KIND_ROCM;

unsafe extern "C" fn device_count() -> u32 {
    0
}

unsafe extern "C" fn device_info(_device: u32, _out: *mut DeviceInfo) -> Status {
    STATUS_INVALID_ARGUMENT
}

unsafe extern "C" fn alloc(_device: u32, _bytes: usize, _out: *mut *mut c_void) -> Status {
    STATUS_UNSUPPORTED
}

unsafe extern "C" fn free(_device: u32, _ptr: *mut c_void) {}

unsafe extern "C" fn copy(_device: u32, _dst: *mut c_void, _src: *const c_void, _bytes: usize) -> Status {
    STATUS_UNSUPPORTED
}

unsafe extern "C" fn supports_op(_op: u32) -> u32 {
    0
}

unsafe extern "C" fn execute(
    _device: u32,
    _op: u32,
    _attrs: *const OpAttrs,
    _inputs: *const TensorDesc,
    _n_inputs: u32,
    _outputs: *const TensorDesc,
    _n_outputs: u32,
) -> Status {
    STATUS_UNSUPPORTED
}

unsafe extern "C" fn synchronize(_device: u32) -> Status {
    STATUS_OK
}

unsafe extern "C" fn last_error() -> *const c_char {
    c"AMD ROCm/HIP backend is not implemented yet".as_ptr()
}

static VTABLE: PluginVTable = PluginVTable {
    abi_version: ABI_VERSION,
    plugin_version: 0,
    name: c"rocm".as_ptr(),
    device_count,
    device_info,
    alloc,
    free,
    copy_from_host: copy,
    copy_to_host: copy,
    copy_device_to_device: copy,
    supports_op,
    execute,
    synchronize,
    last_error,
};

#[unsafe(no_mangle)]
pub extern "C" fn pytorches_plugin_entry() -> *const PluginVTable {
    &VTABLE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_reports_no_devices() {
        let vt = unsafe { &*pytorches_plugin_entry() };
        assert_eq!(vt.abi_version, ABI_VERSION);
        assert_eq!(unsafe { (vt.device_count)() }, 0);
        assert_eq!(unsafe { (vt.supports_op)(op::ADD) }, 0);
    }
}
