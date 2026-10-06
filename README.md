# PyTorches

A PyTorch-interoperable tensor/autograd library in Rust. All device code lives in
**plugins**: shared libraries loaded at runtime from a plugin directory, so the core stays
backend-free and different developers can own different backends.

## Layout

| Path | Role |
|---|---|
| `crates/plugin-abi` | The stable `#[repr(C)]` plugin ABI. The only crate a plugin author needs. |
| `crates/core` | `Tensor`, autograd, device registry, plugin loader. No device code. |
| `crates/python` | PyO3 bindings (`import pytorches`), built with maturin. |
| `plugins/<name>/` | One crate per backend: `pytorches-plugin-<name>` → `pytorches_plugin_<name>.dll/.so/.dylib`. |
| `plugins/bin/` | Built plugin libraries; this is the directory the runtime scans. |
| `tests/diff` | Differential tests against PyTorch. |

## Build and test

```powershell
cargo test                                   # Rust tests (CPU plugin linked statically)
.\scripts\build-plugins.ps1                  # build every plugins/* crate into plugins/bin
python -m venv .venv --system-site-packages  # reuse the installed PyTorch as the oracle
.venv\Scripts\python -m pip install maturin
cd crates\python; ..\..\.venv\Scripts\maturin develop --release; cd ..\..
.venv\Scripts\python -m unittest discover -s tests\diff -v
```

Plugin discovery order: `$PYTORCHES_PLUGIN_DIR` (path list), else `./plugins/bin` and
`<exe dir>/plugins`. Only files named `pytorches_plugin_*` are considered. A plugin that
reports zero devices (hardware/driver absent) is skipped.

```python
import pytorches as pt
pt.devices()                        # ['cpu:0', ...]
x = pt.Tensor([1., 2.], [2], device="cpu:0").to("cpu:0")
```

## Writing a plugin

1. `cargo new --lib plugins/<name>`, set `crate-type = ["cdylib"]`, depend on `pytorches-plugin-abi`
   (or implement the header by hand from `crates/plugin-abi/src/lib.rs`; C/C++ works too).
2. Export `pytorches_plugin_entry() -> *const PluginVTable` returning a static vtable.
   `plugins/cpu/src/lib.rs` is the reference implementation.
3. Implement: `device_count`, `device_info`, `alloc`/`free`, `copy_from_host`/`copy_to_host`,
   `supports_op`, `execute`, `synchronize`, `last_error`.
4. `execute` receives strided input descriptors (stride 0 = broadcast) and contiguous,
   pre-allocated outputs. Shape inference is done by the core. Never unwind across the boundary.
5. Run `scripts/build-plugins.ps1`; the library lands in `plugins/bin`.

ABI changes: ops are append-only; layout changes bump `ABI_VERSION` and old plugins are rejected
at load time with a clear error.

## Status

Milestone 2: plugin architecture. `f32`, contiguous tensors; add/sub/mul/div (broadcasting),
neg/exp/log/relu/tanh, sum/mean, 2-D matmul/transpose, reshape; autograd and device
placement (`.to`) verified against PyTorch via the CPU plugin.

Next: DLPack + safetensors interop, CUDA plugin (RTX / sm_120), Intel Arc plugin (Level Zero),
then memory-aware placement across devices.
