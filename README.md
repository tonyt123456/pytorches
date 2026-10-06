<div align="center">

# PyTorches

**One tensor library. Every GPU. No recompiling.**

A PyTorch-interoperable tensor and autograd engine written in Rust. It detects your hardware
and optimizes for it automatically. Developers can add support for new hardware by writing a
plugin.

![status](https://img.shields.io/badge/status-early%20alpha-orange)
![rust](https://img.shields.io/badge/rust-2024-b7410e)
![license](https://img.shields.io/badge/license-Apache--2.0-blue)

</div>

---

## Why

Today, running PyTorch on your hardware means picking the right build: a CUDA wheel for your
toolkit, a ROCm build, an XPU build. Support for a new accelerator means patching and
recompiling the framework itself, and a laptop with an NVIDIA GPU *and* an Intel GPU can't
use both well.

PyTorches turns that around:

- **Hardware optimization is automatic.** PyTorches detects what's in your machine at startup
  and uses the matching backends. You don't choose a CUDA vs ROCm vs XPU build, and you don't
  configure anything. The same install works on an NVIDIA machine, an Intel Arc machine, or a
  machine with both.
- **The core stays hardware-agnostic.** It owns tensors, autograd and dispatch, and has no list
  of vendors. Each hardware backend is a separate **plugin** behind a small, stable C ABI, and
  each plugin runs its own hardware test, so only the ones that fit your machine are used.
- **Developers can add hardware support without touching the core.** If you build an
  accelerator, or want to tune for one, you write a plugin. NVIDIA, AMD and Intel plugins can
  live in separate repos with separate release cycles. A plugin depends on one tiny
  `#![no_std]` crate (or a header, since C and C++ plugins work too).
- **Heterogeneous machines are the target.** An 8 GB discrete GPU next to an iGPU with 47 GB
  of shared memory should be one pool of devices, not two incompatible installs.

## Architecture

```
            ┌─────────────────────────────────────────────┐
 Python  →  │  pytorches (PyO3)                           │
            ├─────────────────────────────────────────────┤
            │  core: Tensor · autograd · device registry  │   no device code
            ├─────────────────────────────────────────────┤
            │  plugin-abi: stable #[repr(C)] vtable       │   ← the contract
            └───────┬───────────────┬───────────────┬─────┘
                    │ dlopen        │ dlopen        │ dlopen
        plugins/bin/│               │               │
        ┌───────────▼───┐  ┌────────▼──────┐  ┌─────▼─────────┐
        │ plugin_cpu    │  │ plugin_cuda   │  │ plugin_xpu    │  …
        │ (reference)   │  │ (planned)     │  │ (planned)     │
        └───────────────┘  └───────────────┘  └───────────────┘
```

**How hardware gets picked up automatically**

You don't do anything: PyTorches ships with its backends, and at startup it works out which
ones apply to your machine.

1. The core scans the plugin directory for `pytorches_plugin_*` libraries.
2. It loads each one and checks the ABI version.
3. It calls the plugin's `device_count()`. That call is the **hardware test**, and the plugin
   owns it: a CUDA plugin asks the NVIDIA driver, an Intel plugin asks Level Zero, and so on.
4. A plugin that reports zero devices (no hardware, no driver, wrong generation) is
   **unloaded and ignored**. A plugin that reports devices is registered, and its devices
   show up as `cuda:0`, `xpu:0`, and so on.

So the same install works on a machine with an NVIDIA GPU, one with an Arc GPU, or neither,
and each machine ends up with only the backends it can actually use. Because the core never
inspects hardware itself, supporting a new accelerator never touches the core.

**How a plugin works** *(for developers)*

- It exports one symbol, `pytorches_plugin_entry`, returning a static vtable: device
  enumeration, `alloc`/`free`, host copies, `supports_op`, `execute`, `synchronize`.
- The core does shape inference and allocates output buffers; the plugin only fills them.
- Operands arrive as **strided descriptors**. Broadcasting is stride 0 and transposing is a
  strided copy, so backends never implement "broadcast" or "transpose" as special cases.
- The ABI version is checked before anything else. A plugin built for a different ABI is
  rejected with a clear message, and one whose hardware test finds nothing is skipped silently.
- `execute` returns a status code. Nothing Rust-specific (no `Vec`, `String`, `Box`)
  crosses the boundary.

## Quick start

Requires Rust (stable), the MSVC build tools on Windows, and Python 3.10+.

```powershell
git clone https://github.com/tonyt123456/pytorches
cd pytorches

cargo test                                    # core tests (CPU plugin linked statically)
.\scripts\build-plugins.ps1                   # build every plugins/* crate into plugins/bin

python -m venv .venv --system-site-packages
.venv\Scripts\python -m pip install maturin
cd crates\python; ..\..\.venv\Scripts\maturin develop --release; cd ..\..
```

Then fit a line with autograd:

```python
import pytorches as pt

print(pt.devices())                 # ['cpu:0'] + whatever plugins are in plugins/bin

xs = [i / 10 for i in range(20)]
x = pt.Tensor(xs, [20, 1])
y = pt.Tensor([3 * v + 1 for v in xs], [20, 1])      # target: y = 3x + 1

w = pt.Tensor([0.0], [1, 1], requires_grad=True)
b = pt.Tensor([0.0], [1], requires_grad=True)

for _ in range(300):
    w.zero_grad(); b.zero_grad()
    err = x @ w + b - y
    loss = (err * err).mean()
    loss.backward()
    # no optimizers yet: take the step by hand and re-wrap as fresh leaves
    w = pt.Tensor((w - 0.1 * w.grad).tolist(), [1, 1], requires_grad=True)
    b = pt.Tensor((b - 0.1 * b.grad).tolist(), [1], requires_grad=True)

print(w.item(), b.item())           # ≈ 3.000, 1.000
```

Move work between devices; the move is differentiable:

```python
t = pt.Tensor([1.0, 2.0], [2], device="cpu:0")
t.to("cuda:0")                      # available once the CUDA backend lands (planned)
```

*Advanced:* plugins are found in `$PYTORCHES_PLUGIN_DIR`, else `./plugins/bin`, else
`<exe dir>/plugins`. Only files named `pytorches_plugin_*` are considered.

## Correctness first

Every op is checked against PyTorch itself. `tests/diff` runs each op in both libraries on
random inputs (including broadcasting shapes) and compares the **forward values and the
gradients**. A new backend inherits that oracle for free.

```powershell
.venv\Scripts\python -m unittest discover -s tests\diff -v
```

## For developers: writing a plugin

Most users never need this. It's for people adding support for new hardware, or tuning for
existing hardware, without touching the core.

1. `cargo new --lib plugins/<name>`, set `crate-type = ["cdylib"]`, depend on
   [`pytorches-plugin-abi`](crates/plugin-abi/src/lib.rs). The crate is the full contract
   and is documented inline. Implementing the same struct from C works too.
2. Export `pytorches_plugin_entry() -> *const PluginVTable`.
3. Implement the vtable. [`plugins/cpu`](plugins/cpu/src/lib.rs) is a complete reference
   in about 270 lines.
4. `.\scripts\build-plugins.ps1` copies `pytorches_plugin_<name>.dll` into `plugins/bin`.
5. Run the differential tests. If they pass, your backend is correct.

ABI evolution: ops are append-only. A layout change bumps `ABI_VERSION`.

## Repository layout

| Path | What it is |
|---|---|
| `crates/plugin-abi` | The stable plugin ABI. The only crate a plugin author needs. |
| `crates/core` | `Tensor`, reverse-mode autograd, device registry, plugin loader. |
| `crates/python` | PyO3 bindings, built with maturin. |
| `plugins/<name>` | One crate per backend. `cpu` is the reference. |
| `plugins/bin` | Built plugin libraries; the runtime scans this directory. |
| `tests/diff` | Differential tests against PyTorch. |

## Status and roadmap

PyTorches is **early alpha**. It is not a PyTorch replacement yet, and it is not fast yet.
The CPU plugin is a naive reference implementation.

**Working today**

- `f32` tensors; add/sub/mul/div with broadcasting; neg, exp, log, relu, tanh
- sum, mean, 2-D matmul, transpose, reshape
- Reverse-mode autograd (gradient accumulation, shared subexpressions, broadcast reduction)
- Runtime plugin loading, device enumeration, differentiable cross-device `.to()`
- Python bindings and a differential test suite against PyTorch

**Next**

- [ ] CUDA plugin (NVIDIA, targeting current toolkits and Blackwell via PTX JIT)
- [ ] Intel Arc plugin (Level Zero / SPIR-V)
- [ ] ROCm plugin
- [ ] DLPack zero-copy exchange with `torch.Tensor`; `safetensors` and `state_dict` loading
- [ ] Memory-aware placement: run what fits on the fast GPU, spill or offload the rest to the big
  one, with transfer cost in the model
- [ ] Strided views, more dtypes (f16/bf16/i64), more ops, `nn` and optimizers
- [ ] Fusion and tuned matmul/attention kernels
- [ ] Hardware detection plus on-demand plugin download
- [ ] Intel NPU plugin (inference only: static-shape graphs)

**Non-goals (for now):** matching PyTorch's 2,000+ ops, or supporting old GPU generations.
Targeting only current drivers and toolkits is what keeps the project tractable.

## Contributing

The plugin boundary is the main way to contribute: a new backend needs no change to the core.
Issues and PRs are welcome. Run `cargo test` and the differential tests before submitting.

## License

Apache-2.0. See [LICENSE](LICENSE).
