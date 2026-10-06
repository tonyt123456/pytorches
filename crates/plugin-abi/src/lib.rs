//! The stable C ABI between the PyTorches core and device plugins.
//!
//! A plugin is a shared library named `pytorches_plugin_<name>.{dll,so,dylib}` that
//! exports one symbol, [`ENTRY_SYMBOL`], returning a pointer to a static
//! [`PluginVTable`]. Plugin authors depend only on this crate. Nothing Rust-specific
//! (no `Vec`, `String`, `Box`, trait objects) crosses the boundary, so plugins may be
//! built with any compiler version, or in C/C++.
//!
//! # Contract
//! * The core checks `abi_version` (the first field) before touching anything else.
//! * Plugins own their device memory. The core only holds opaque pointers and calls
//!   `alloc`/`free`/`copy_*`. Cross-device copies are staged through host memory by the core.
//! * `execute` runs one op. The core allocates the output buffers (via `alloc`) and does
//!   shape inference; the plugin only fills the outputs. Inputs may be strided (including
//!   stride 0 for broadcasting), outputs are always contiguous.
//! * `execute` must not panic/unwind across the boundary; return a [`Status`] instead.
//! * All entry points may be called from any thread.
//! * Execution model: every call enqueues work on the device's implicit default stream, in
//!   order. `execute` and the `copy_*` entry points that write device memory may return before
//!   the work finishes. `copy_to_host` and `synchronize` block until the data/device is ready.
//!   `free` must be safe while earlier work that uses the buffer is still in flight.
//!   (Explicit streams can be added later as appended fields.)
//!
//! Adding ops or fields in a backward-compatible way means appending to the op list or
//! bumping `ABI_VERSION`; plugins advertise which ops they implement via `supports_op`.
#![no_std]

use core::ffi::{c_char, c_void};

pub const ABI_VERSION: u32 = 2;
/// NUL-terminated name of the exported entry function (`EntryFn`).
pub const ENTRY_SYMBOL: &[u8] = b"pytorches_plugin_entry\0";
/// Required file-name prefix for plugin libraries in the plugin directory.
pub const FILE_PREFIX: &str = "pytorches_plugin_";

pub type Status = i32;
pub const STATUS_OK: Status = 0;
pub const STATUS_UNSUPPORTED: Status = 1;
pub const STATUS_OUT_OF_MEMORY: Status = 2;
pub const STATUS_INVALID_ARGUMENT: Status = 3;
pub const STATUS_INTERNAL: Status = 4;

pub const DTYPE_F32: u32 = 0;

pub const KIND_CPU: u32 = 0;
pub const KIND_CUDA: u32 = 1;
pub const KIND_ROCM: u32 = 2;
pub const KIND_XPU: u32 = 3;
pub const KIND_NPU: u32 = 4;
pub const KIND_OTHER: u32 = 255;

/// Reported by `device_info` when a figure is unavailable.
pub const MEMORY_UNKNOWN: u64 = u64::MAX;

/// Op codes passed to `execute`. Append only; never renumber.
pub mod op {
    pub const NEG: u32 = 1;
    pub const EXP: u32 = 2;
    pub const LOG: u32 = 3;
    pub const RELU: u32 = 4;
    pub const TANH: u32 = 5;
    /// 1.0 where x > 0 else 0.0 (relu derivative).
    pub const STEP: u32 = 6;

    /// Elementwise binary ops. Inputs have the output's shape; broadcast inputs carry stride 0.
    pub const ADD: u32 = 100;
    pub const SUB: u32 = 101;
    pub const MUL: u32 = 102;
    pub const DIV: u32 = 103;

    /// Row-major `[m,k] x [k,n] -> [m,n]`, contiguous inputs.
    pub const MATMUL: u32 = 200;
    /// Sum over axis `attrs.ints[0]`, removing it; contiguous input.
    pub const SUM_AXIS: u32 = 201;

    /// Materialize a (possibly strided / broadcast) input into a contiguous output.
    pub const COPY: u32 = 300;

    /// Fill the output with a constant. No inputs. `attrs.ints[0]` is the f32 bit pattern
    /// (`f32::to_bits() as i64`).
    pub const FILL: u32 = 400;
    /// Fill the output with N(0,1) samples. No inputs. `attrs.ints[0]` is the seed.
    ///
    /// Counter-based so every device can produce the same stream: for element index `i` (u64):
    /// `h = splitmix64(seed as u64 ^ i.wrapping_mul(0x9E3779B97F4A7C15))`,
    /// `u1 = ((h >> 40) + 1) as f32 / 16777216.0` (in (0,1]),
    /// `u2 = (h & 0xFFFFFF) as f32 / 16777216.0` (in [0,1)),
    /// `out = sqrt(-2 ln u1) * cos(2 pi u2)`.
    /// where `splitmix64(x)`: `x += 0x9E3779B97F4A7C15; x = (x ^ (x>>30)) * 0xBF58476D1CE4E5B9;
    /// x = (x ^ (x>>27)) * 0x94D049BB133111EB; x ^ (x>>31)` (wrapping u64 arithmetic).
    pub const RAND_NORMAL: u32 = 401;
}

/// A tensor argument. `shape` and `strides` point to `ndim` entries; strides are in elements.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct TensorDesc {
    pub data: *mut c_void,
    pub dtype: u32,
    pub ndim: u32,
    pub shape: *const u64,
    pub strides: *const u64,
}

/// Small fixed attribute block for ops that need parameters.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct OpAttrs {
    pub ints: [i64; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DeviceInfo {
    /// NUL-terminated device name.
    pub name: [c_char; 64],
    pub kind: u32,
    pub total_memory: u64,
    pub free_memory: u64,
}

#[repr(C)]
pub struct PluginVTable {
    /// Must be [`ABI_VERSION`]. Keep this the first field.
    pub abi_version: u32,
    pub plugin_version: u32,
    /// Static NUL-terminated plugin name, e.g. `"cpu"`, `"cuda"`. Used in device strings (`cuda:0`).
    pub name: *const c_char,

    /// Number of usable devices; 0 if the hardware/driver is absent.
    pub device_count: unsafe extern "C" fn() -> u32,
    pub device_info: unsafe extern "C" fn(device: u32, out: *mut DeviceInfo) -> Status,

    pub alloc: unsafe extern "C" fn(device: u32, bytes: usize, out: *mut *mut c_void) -> Status,
    pub free: unsafe extern "C" fn(device: u32, ptr: *mut c_void),
    pub copy_from_host:
        unsafe extern "C" fn(device: u32, dst: *mut c_void, src: *const c_void, bytes: usize) -> Status,
    pub copy_to_host:
        unsafe extern "C" fn(device: u32, dst: *mut c_void, src: *const c_void, bytes: usize) -> Status,
    /// Copy within one device (both pointers from this plugin and `device`). Ranges must not overlap.
    pub copy_device_to_device:
        unsafe extern "C" fn(device: u32, dst: *mut c_void, src: *const c_void, bytes: usize) -> Status,

    /// Nonzero if `execute` implements `op`.
    pub supports_op: unsafe extern "C" fn(op: u32) -> u32,
    pub execute: unsafe extern "C" fn(
        device: u32,
        op: u32,
        attrs: *const OpAttrs,
        inputs: *const TensorDesc,
        n_inputs: u32,
        outputs: *const TensorDesc,
        n_outputs: u32,
    ) -> Status,
    /// Blocks until all queued work on `device` is done.
    pub synchronize: unsafe extern "C" fn(device: u32) -> Status,
    /// Static or thread-local NUL-terminated message for the last failure on this thread.
    pub last_error: unsafe extern "C" fn() -> *const c_char,
}

// The vtable is immutable static data (function pointers + a static string).
unsafe impl Sync for PluginVTable {}

pub type EntryFn = unsafe extern "C" fn() -> *const PluginVTable;
