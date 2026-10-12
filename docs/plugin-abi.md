# The plugin contract

This document defines the line between the PyTorches core and a device plugin: what each side owns, what
a plugin must do to be called conforming, and how the contract changes over time. The Rust definitions
in [`crates/plugin-abi/src/lib.rs`](../crates/plugin-abi/src/lib.rs) are the machine-readable half; this
is the normative prose.

**Status: ABI 4, alpha.** The contract below describes what the core and the three existing plugins (CPU,
CUDA, Intel GPU) actually do. It is not yet frozen; the open questions in the last section must be settled
before a 1.0 ABI. Every rule carries a tag saying how strongly it is backed today:

- **[enforced]** the core checks it and refuses or fails if it is violated
- **[tested]** the differential tests or a plugin's conformance tests exercise it, but nothing stops a
  plugin from violating it
- **[convention]** the existing plugins follow it and the core relies on it, with no check
- **[proposed]** not true of the code today; this is the rule going forward

The words MUST, SHOULD and MAY are used as in RFC 2119.

---

## 1. Who owns what

The core is hardware-agnostic. It has no list of vendors and never inspects hardware. A plugin knows its
hardware and nothing about tensors, autograd or models.

| Concern | Owner | Notes |
|---|---|---|
| Tensors, shapes, strides, views | core | A plugin sees a buffer handle plus a shape/stride description per call |
| Autograd, graph construction, backward rules | core | Plugins never see a graph |
| Shape inference and output allocation | core | The core asks the plugin for the output buffer; the plugin only fills it |
| Which device runs an op | core | The planner and the user decide; a plugin is only ever asked to run ops on its own devices |
| Placement, splitting, cost model, device profiling | core | Uses `device_info` and measured throughput through ordinary ops |
| Cross-device copies | core | Staged through host memory with `copy_to_host` / `copy_from_host`; peer copies are not in the ABI |
| Device memory: allocation, pooling, reuse, OOM handling | plugin | The core holds opaque handles and never looks inside |
| Executing ops, including kernel choice, tiling, scratch buffers | plugin | See section 6 |
| Synchronization and stream/queue management | plugin | The core sees "enqueue" and "wait" (section 5) |
| Hardware detection, driver and runtime loading | plugin | Lazily, inside the plugin; a missing driver means zero devices, not a load failure |
| Packaging, versioning, release cadence of a backend | plugin author | A backend is its own package (section 2) |

A plugin MAY make any internal decision that does not change the result of an op: choose kernels per shape,
materialize transposes, run an op as several kernels, cache buffers, reorder independent work on its
own queue. The Arc plugin, for example, computes `A B^T` as `(B A^T)^T` when that moves fewer bytes. That
is the plugin's business, and the core cannot tell.

A plugin MUST NOT assume anything about the core: no global state shared with it, no retaining of pointers
the core passes in beyond the call (descriptors and attribute blocks are only valid during the call),
no calling back into the core. **[convention]**

## 2. Packaging and discovery

A plugin is a shared library exporting one symbol. **[enforced]**

- File name `pytorches_plugin_<name>.{dll,so,dylib}`; the loader ignores anything else in a plugin directory.
- Entry symbol `pytorches_plugin_entry`, of type `extern "C" fn() -> *const PluginVTable`, returning a
  pointer to a static table that lives as long as the library. Plugins are never unloaded.
- `<name>` is the plugin's name (the vtable `name`), lowercase, and is also the device type in device
  strings (`cuda:0`). Two plugins with the same name: the first one loaded wins.

A standalone backend ships as a Python package containing the library and a manifest, and declares an
entry point in the `pytorches.plugins` group naming a module. The core locates that module's directory
without importing it. **[enforced]**

**Manifest** `pytorches_plugin_<name>.json` next to the library, optional. Recognized keys: `name`, `vendor`,
`abi`, `os`, `arch`. Unknown keys are ignored. A manifest that does not match the running core (ABI number,
OS, CPU architecture) causes the plugin to be skipped before any of its code runs. **[enforced]**

**Loading gates**, in order, for every candidate library: manifest, then probe, then load. **[enforced]**

1. *Manifest check* as above.
2. *Probe*: the host loads the library in a short-lived child process with a timeout, calls
   `device_count` and `device_info`, and discards the process. A crash, a hang, a bad table, or zero
   devices skips the plugin. Results are cached on disk (24 hours for success, 10 minutes for failure,
   keyed by file identity). A plugin therefore MUST be safe to load and query in a process that exits
   immediately afterwards, and its probe path MUST NOT depend on state a previous process left behind.
3. *Load*: the host loads the library and re-checks the table and device count.

A plugin MUST NOT print to stdout during `pytorches_plugin_entry`, `device_count` or `device_info`:
the probe protocol reads stdout. (stderr is fine.) **[enforced by the probe parser ignoring untagged
lines; keep it quiet anyway]**

## 3. The vtable and versioning

`PluginVTable` starts with two `u32` fields, in this order, in every ABI major version: `abi_version` and
`struct_size`. **[enforced]**

- `abi_version` is the **major** version. The core rejects a plugin whose major version differs from its
  own. A major bump means a breaking layout or semantic change.
- `struct_size` is `size_of::<PluginVTable>()` at the plugin's build time. The core rejects a table smaller
  than the fields it needs, and accepts a larger one. A plugin built against a newer header than the core
  therefore loads, and the core simply never reads the extra fields.
- Compatible additions are **appended** fields and **appended** op codes. They do not change
  `abi_version`. Fields are never removed, reordered or resized, and op codes are never renumbered or reused.
  A field the core wants to read that lies beyond a plugin's `struct_size` MUST be treated by the core as
  "not implemented" (this is how a future optional entry point would be introduced). **[proposed: no
  appended field exists yet, so this path is untested]**
- `plugin_version` is the plugin's own version number, informational.

The ABI contains no Rust types: only integers, pointers and `#[repr(C)]` structs. A plugin may be written
in any language. **[enforced by construction]**

## 4. Devices and memory

**Devices.** `device_count()` returns the number of usable devices, `0` if the hardware or its driver is
absent. It is the plugin's hardware test and MUST NOT fail, abort or block indefinitely when the hardware is
missing. Device indices `0..count` are stable for the life of the process. `device_info(i, out)` fills the
name (NUL-terminated, at most 63 characters), a `kind` constant, total and free memory in bytes
(`MEMORY_UNKNOWN` when it cannot be known) and `DEVICE_FLAG_*` bits. **[enforced]**

- `kind` is one of the `KIND_*` constants and is descriptive; the core does not branch on it.
- `DEVICE_FLAG_SHARED_HOST_MEMORY` MUST be set for a device that allocates from system RAM (integrated
  GPUs, unified memory). The planner caps such a device's free memory at the host's and accounts for the
  pool being shared with the CPU. **[enforced: the planner reads it]**
- `free_memory` is advisory. The core uses it to rank placements and MUST treat an `alloc` failure as the
  truth; a plugin whose figure is an estimate says so in its documentation.

**Allocation.** `alloc(device, bytes, &mut ptr)` returns an opaque handle in `ptr`. **[enforced]**

- The handle is opaque: it may be a device pointer, a driver object handle, or anything else that fits in a
  pointer. The core stores, compares and passes it back; it never dereferences it or does arithmetic on it.
  Consequently a buffer is always addressed from its start, and views are expressed only through
  shape/strides, never through a base offset. A plugin MUST NOT require the core to pass interior pointers.
- A returned buffer MUST be usable for a vectorized access of 16 bytes at offset 0 (alignment of at least 16;
  64 is recommended and is what CPU, CUDA and OpenCL runtimes give in practice). Plugins already rely on
  this for wide loads. **[convention]**
- `alloc` of a size the device cannot provide MUST return `STATUS_OUT_OF_MEMORY` and leave the plugin in a
  usable state; it MUST NOT abort or panic. A request for more than the device's total memory MUST fail the
  same way, quickly. **[tested]** (every plugin has an out-of-memory test)
- A plugin that pools buffers MUST release its cache and retry before reporting out-of-memory, so pooling
  never turns a request that would fit into a failure. **[convention]**
- Contents of a new buffer are undefined. Sizes are in bytes; a zero-byte request returns a valid handle.

**Free.** `free(device, ptr)` returns the buffer. It MUST be safe to call while queued work that uses the
buffer is still running (section 5). A plugin that reuses the memory MUST order the reuse after that work.
**[convention]**

**Copies.** `copy_from_host`, `copy_to_host` and `copy_device_to_device` move `bytes` bytes between a plain
host pointer and a buffer, or between two buffers of the same device. The device-to-device ranges never
overlap. There are no sub-range offsets: copies address buffer starts. (Sub-range copies are an open item.)
**[enforced]**

## 5. Execution model

Every call to a plugin enqueues work on that device's implicit default stream, **in order**. **[convention]**

- `execute` and the copies that write device memory MAY return before the work finishes.
- `copy_to_host` returns only when the data has arrived in host memory. `synchronize` returns only when all
  previously enqueued work on the device is complete. These are the only blocking calls the core relies on.
- Results of one call are visible to the next call on the same device without any further synchronization.
- A plugin MUST NOT require the core to call `synchronize` for correctness of a sequence of ops on one
  device; `synchronize` exists for timing, for reading results, and for surfacing deferred errors.
- Work on different devices, even of the same plugin, is unordered. The core orders cross-device
  transfers itself (`copy_to_host` blocks).

A deferred failure (for example, a kernel that faults after `execute` returned `STATUS_OK`) MUST be
reported by the next call that can report it, usually `synchronize` or a blocking copy. **[convention]**

Explicit streams, events and overlap of transfers with compute are not part of ABI 4 (section 11).

## 6. Tensors and ops

**Descriptors.** An operand is a `TensorDesc`: opaque `data` handle, `dtype`, `ndim`, and `shape` and
`strides` arrays of `ndim` entries; strides are in **elements**. **[enforced]**

- Inputs MAY be strided, including stride 0 (broadcast) and strides that make a transposed view. A plugin
  MUST handle any input layout the core can produce, or reject it with `STATUS_UNSUPPORTED`. In
  particular, broadcasting and transposition are layouts, never ops: a plugin has no "broadcast" or
  "transpose" op to implement. (The matmul family below has stricter input rules.)
- The output is **always contiguous row-major**, with the shape the core inferred. The core allocates it.
- `ndim` may be `0` (a scalar). Zero-element tensors are possible; a plugin MUST accept them and do nothing.
- The only dtype is `DTYPE_F32`. A plugin MUST reject other dtypes with `STATUS_UNSUPPORTED`. **[enforced by
  the core only ever sending f32; see section 11]**

**Ops are tiered.**

*Required.* Every plugin MUST implement these; the core calls them without asking `supports_op`, and an
unimplemented required op fails the call. `supports_op` MUST return nonzero for each. **[enforced: the core
asserts support before every call]**

| Op | Meaning |
|---|---|
| `NEG EXP LOG RELU TANH STEP` | elementwise unary; `STEP` is `1.0` where `x > 0` else `0.0` |
| `ADD SUB MUL DIV` | elementwise binary; operands already have the output's shape, broadcast inputs carry stride 0 |
| `MATMUL` | row-major `[m,k] x [k,n] -> [m,n]`, contiguous inputs |
| `SUM_AXIS` | sum over axis `attrs.ints[0]`, removing it; contiguous input |
| `COPY` | materialize a (possibly strided or broadcast) input into a contiguous output |
| `FILL` | fill the output with the constant whose f32 bit pattern is `attrs.ints[0]` |
| `RAND_NORMAL` | counter-based N(0,1) fill, specified exactly in the ABI header, so every device produces the same stream |

*Optional, with a defined core fallback.* A plugin MAY implement these. The core asks `supports_op` and,
when the answer is zero, performs the documented fallback with required ops. A plugin SHOULD implement an
optional op only when it beats the fallback; it MAY decline one it cannot do better. **[enforced]**

| Op | Meaning | Core fallback |
|---|---|---|
| `MATMUL_T` | matmul over stored contiguous operands, `attrs.ints[0]` bit 0 = use `A^T`, bit 1 = use `B^T` | materialize the transposes with `COPY`, then `MATMUL` |
| `AXPY` | `out = in0 + alpha * in1` (alpha is an f32 bit pattern in `attrs.ints[0]`), `out` MAY alias `in0` | `MUL`, `ADD` and `COPY` |

*Extension ops* (future). New op codes are appended. A new op is introduced as optional with a fallback
defined at the same time, so that an older plugin remains conforming. An op that has no sensible fallback
cannot be added without a major version bump. **[proposed]**

**Numerics.** The CPU plugin is the reference. A conforming plugin's results MUST agree with it to within
float rounding of the operation: elementwise ops to a few ulps, reductions and matmul to a relative error
that scales with the contracted size (the existing tests use `1e-5` for transcendental ops and `2e-4` for
matmul at sizes up to a few thousand). Summation order is the plugin's choice. `RAND_NORMAL` is specified
bit-for-bit up to the accuracy of `log`, `sqrt` and `cos`. NaN and infinity propagate as IEEE-754 does.
**[tested]**

**Attributes.** `OpAttrs` is a fixed block of four `i64`s; ops that need more parameters get a new op
code, not a bigger block, until the ABI grows an extension mechanism (section 11).

## 7. Errors

Every entry point that can fail returns a `Status`: `STATUS_OK`, `STATUS_UNSUPPORTED`,
`STATUS_OUT_OF_MEMORY`, `STATUS_INVALID_ARGUMENT` or `STATUS_INTERNAL`. **[enforced]**

- On failure `last_error()` MUST return a NUL-terminated message for the failure on the **calling thread**,
  valid until the next call into the plugin from that thread. Messages SHOULD name the operation and the
  numbers involved ("cuMemAlloc of 68719476736 bytes: out of memory").
- A plugin MUST NOT panic, throw or otherwise unwind across the boundary, abort the process, or leave
  itself unusable after a returned error. Any exception in the plugin's own language MUST be caught and
  converted to `STATUS_INTERNAL`. **[convention]; tested by each plugin's out-of-memory test**
- The core maps `STATUS_OUT_OF_MEMORY` to its out-of-memory error (a Python `MemoryError`, and the trigger
  for `pt.place`'s fall-back to the next device) and the others to invalid/runtime errors.
  `STATUS_UNSUPPORTED` and `STATUS_INVALID_ARGUMENT` currently surface the same way.

## 8. Threading

Every entry point MAY be called concurrently from any number of threads. **[convention]**

- A plugin MUST protect its own state. Calls into one device from several threads behave as if serialized
  in some order; the "in-order stream" of section 5 is the order in which the calls took effect.
- `last_error` is per thread.
- The core makes no ordering promises between threads: two threads running ops on the same device may
  interleave in any order, and the plugin is not required to make that deterministic.

## 9. What a plugin author does, and what makes a plugin conforming

1. Depend on the `pytorches-plugin-abi` crate (or the header generated from it **[proposed]**), implement
   the vtable, export the entry symbol.
2. Implement every required op and `supports_op` honestly. Implement optional ops only when they are
   faster than the fallback.
3. Return `device_count() == 0`, quietly, when the hardware is absent.
4. Add a manifest and, to ship separately, an entry point (section 2).
5. Pass the conformance checks below.

**Conformance, today.** There is no single plugin-agnostic suite yet; conformance is the union of:

- the differential tests in `tests/diff`, which run every op against PyTorch on every detected device and
  compare forward values and gradients **[tested]**;
- each existing plugin's own conformance tests, which compare it with the CPU plugin through the core:
  odd shapes, broadcasting and strided inputs, in-place `AXPY`, transposed matmul, out-of-memory behavior,
  and a bound on allocation count and size **[tested]**.

**Conformance, intended.** A plugin-agnostic harness, `pytorches-conformance`, that loads one library and
runs the whole contract against it with no dependence on PyTorch: every required and every advertised
optional op over a grid of shapes and layouts (scalar, empty, broadcast, transposed, non-multiple-of-tile
sizes, sizes straddling 2^31 elements when memory allows) against a host reference; allocation edge cases
and out-of-memory recovery; the error rules of section 7; concurrent calls from many threads; and the
probe protocol (silent load, zero devices without hardware). Passing it is the definition of
"PyTorches plugin" and earns a badge. **[proposed]**

## 10. Changing the contract

- A change to this document that alters what a plugin must do bumps `abi_version` unless it only adds
  optional behavior.
- Additions follow section 3 (appended fields, appended ops with a fallback).
- Deprecations are announced here for at least one minor release of the core before removal, and removal
  is a major bump.
- The core and the ABI crate are versioned together in this repository; a backend's own releases are
  independent of the core's.

## 11. Open questions (settle before a 1.0 ABI)

These change the contract, so they are listed instead of guessed at.

- **Streams and events.** ABI 4 has one implicit in-order stream per device, so transfers cannot overlap
  compute and two devices of one plugin cannot run concurrently under one host thread of control. Adding
  stream and event handles changes `alloc`/`free` (a stream or fence argument) and every enqueue call.
- **Pointer export.** DLPack for GPU tensors needs a way to ask a plugin for a raw device pointer plus a way
  to wait for pending work. Until then GPU tensors export through a host copy.
- **dtypes.** Only `f32`. Adding f16/bf16/integers/bool needs a per-op dtype support query and decisions
  about mixed-precision accumulation.
- **Per-op or graph-level ops.** Fused ops (a relu gradient, bias plus activation) cut launches, but each
  one in the ABI is a permanent obligation on every plugin. The choice is between a curated set of fused
  ops with fallbacks, as for `AXPY`, and an entry point that takes a small graph.
- **Sub-range copies and peer copies.** Copies address buffer starts only; moving a slice currently needs a
  strided `COPY`. Two devices that can reach each other's memory still stage through host RAM.
- **An attribute extension mechanism** for ops that need more than four integers.
- **Capability query.** `supports_op` answers per op; dtype, maximum allocation, pooling behavior and
  limits that the planner could use are not queryable.
- **A generated C header** (cbindgen) so C and C++ plugin authors do not hand-copy the structs.
