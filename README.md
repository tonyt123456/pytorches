<div align="center">

# PyTorches

**One tensor library. Every GPU. No recompiling.**

A PyTorch-interoperable tensor and autograd engine written in Rust. It detects your hardware
and optimizes for it automatically. It uses every GPU in the machine together: it places work
by the memory each device has free, and splits a model across devices when it won't fit on one.
Developers can add support for new hardware by writing a plugin.

[![PyPI](https://img.shields.io/pypi/v/pytorches-universe)](https://pypi.org/project/pytorches-universe/)
![python](https://img.shields.io/pypi/pyversions/pytorches-universe)
![status](https://img.shields.io/badge/status-early%20alpha-orange)
![license](https://img.shields.io/badge/license-Apache--2.0-blue)

</div>

---

## Install

```
pip install pytorches-universe              # core + CPU backend
pip install "pytorches-universe[cuda]"      # + NVIDIA GPUs  (pytorches-cuda)
pip install "pytorches-universe[xpu]"       # + Intel GPUs   (pytorches-xpu)
pip install "pytorches-universe[gpu]"       # both
```

The package is called `pytorches-universe` on PyPI; you import it as `pytorches`. Then check what it
found on your machine:

```
python -c "import pytorches as pt; pt.doctor()"
```

- **Python 3.11+.** Wheels are published for Windows (x64), Linux (x86_64) and macOS (Apple silicon).
- **No toolkits to install.** The NVIDIA backend needs only the NVIDIA driver (no CUDA toolkit); the Intel
  backend uses the OpenCL runtime that ships with Intel's graphics driver.
- **GPU backends are separate packages**, so you install only what your machine has. Each one is found
  automatically the next time you `import pytorches`; there is nothing to configure or call.
- **Where it runs today:** CPU everywhere; NVIDIA (`pytorches-cuda`) and Intel GPUs (`pytorches-xpu`) on
  Windows and Linux x86_64. AMD (ROCm) and Apple (Metal) backends are planned. The project is developed and
  tested on Windows; the Linux and macOS wheels are built in CI but have had less use.

| Package | What it adds | Source |
|---|---|---|
| `pytorches-universe` | Tensors, autograd, placement planner, CPU backend | [this repo](https://github.com/tonyt123456/pytorches) |
| `pytorches-cuda` | NVIDIA GPUs through the CUDA driver API | [repo](https://github.com/tonyt123456/pytorches-cuda) |
| `pytorches-xpu` | Intel Arc and integrated GPUs through OpenCL | [repo](https://github.com/tonyt123456/pytorches-xpu) |

## Quick start

Train a small MLP. Nothing in the script names a device:

```python
import pytorches as pt

pt.doctor()                                   # what was found, which backends loaded

model = pt.nn.mlp([64, 256, 256, 1])          # Linear -> ReLU stack
x, y = pt.randn([128, 64], seed=1), pt.randn([128, 1], seed=2)
opt = pt.optim.SGD(model.parameters(), lr=0.01)

for step in range(20):
    opt.zero_grad()
    loss = pt.nn.mse_loss(model(x), y)
    loss.backward()
    opt.step()
print(loss.item())
```

Let PyTorches pick the device for a workload, and ask it to explain the choice:

```python
need = pt.nn.estimate_mlp_training_bytes([8192] * 13, batch=32)
plan = pt.plan(need)                          # profiles devices, checks free memory
print(plan)                                   # table + the reason
model = pt.nn.mlp([8192] * 13, device=plan.device)

# or let it build on the best device and fall back if that one runs out of memory:
device, model = pt.place(lambda dev: pt.nn.mlp([8192] * 13, dev), need)
```

Tensors can also be moved explicitly, and the move is differentiable:

```python
t = pt.Tensor([1.0, 2.0], [2], device="cpu:0")
t.to("cuda:0")                                # any device from pt.devices()
```

### If a GPU isn't used

Run `pt.doctor()`. It detects GPUs from the operating system, independent of any plugin, and tells you
for each one whether its backend is loaded, found but skipped (with the reason), or not installed:

```
hardware:
  ok       intel    Intel(R) Arc(TM) 140T GPU (32GB)             'xpu' plugin loaded
  missing  nvidia   NVIDIA RTX PRO 1000 Blackwell Generation Lap no 'cuda' plugin installed (by convention, package pytorches-cuda)
```

Install the missing package and import again. A backend whose driver is broken, hangs, or finds no devices
is skipped without affecting your program (see [How plugins are loaded](#how-plugins-are-loaded)).

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
  accelerator, or want to tune for one, you write a plugin. NVIDIA, AMD and Intel plugins live in
  separate repos with separate release cycles. A plugin depends on one tiny
  `#![no_std]` crate (or a header, since C and C++ plugins work too).
- **Heterogeneous machines are the target.** An 8 GB discrete GPU next to an iGPU with 47 GB
  of shared memory is one pool of devices, not two incompatible installs. Out of the box,
  with no configuration:
  - **Multiple GPUs work together.** NVIDIA and Intel GPUs (and the CPU) are all detected and
    usable in one process, from one install.
  - **Work is placed by available memory.** The planner reads each device's free memory and
    measured speed, puts a workload on the fastest device it fits on, and falls back to a
    bigger one when it doesn't. It accounts for GPUs that share system RAM, so an iGPU is never
    promised more memory than the host really has free.
  - **Work is split between devices.** A model too big for the fast GPU is divided by layers
    across devices, each layer going where it runs best, with activations and gradients moved
    between them automatically. The result matches a single-device run. A model that fits no
    single device can still train.
  - **You can see and steer the decision.** Printing a plan shows which layers go where, what
    moves between devices, and the estimated cost. Placement is a set of pluggable strategies,
    so you can force one (`strategy="layer_split"`) when you know better.

## Works with PyTorch

Tensors and checkpoints move between the two libraries, and every claim below is tested against the
real PyTorch (and the reference `safetensors` package) in
[`tests/diff/test_interop.py`](https://github.com/tonyt123456/pytorches/blob/main/tests/diff/test_interop.py).

```python
import torch, pytorches as pt

# DLPack: zero-copy on the CPU, in both directions (shared memory, safe lifetimes)
t = pt.from_torch(torch_tensor)          # or pt.from_dlpack(any_dlpack_object)
x = pt.to_torch(t)                       # or torch.from_dlpack(t)

# Checkpoints: PyTorch's key names and layouts, so weights load straight across
model = pt.nn.mlp([784, 256, 10])
model.load_state_dict(pt.from_torch_state_dict(torch_model.state_dict()))   # from torch
torch_model.load_state_dict({k: pt.to_torch(v) for k, v in model.state_dict().items()})  # to torch

# safetensors (any of F32/F16/BF16/F64/ints/bool, loaded as float32) and torch.save files
weights = pt.safetensors.load("model.safetensors", device="cuda:0")
pt.safetensors.save(model.state_dict(), "out.safetensors")
state = pt.load_torch("model.pt")        # no torch needed; a restricted unpickler never runs code
```

- **DLPack:** float32 CPU tensors are shared with no copy; other dtypes (f16, bf16, f64, ints, bool) and
  strided views are converted with one copy. A producer on another device (for example PyTorch XPU) is
  asked to copy to the CPU first. Exporting a tensor that lives on a GPU raises a clear error telling you to
  `.to("cpu:0")` first: zero-copy GPU sharing needs a pointer-export hook in the plugin ABI (planned).
- **Checkpoints:** `Linear.weight` is exported as `[out, in]` like `torch.nn.Linear`, and `Sequential`
  children are keyed by index (`0.weight`, `2.bias`). A PyTorch `nn.Sequential(Linear, ReLU, Linear)` checkpoint
  loads into `pt.nn.mlp` and produces the same outputs, on every device.
- **`torch.save` files** are read with an allow-list unpickler: only tensors, dicts and plain values are
  accepted, so a malicious checkpoint is rejected rather than executed (this is tested).

## See it work

[`examples/demo.py`](https://github.com/tonyt123456/pytorches/blob/main/examples/demo.py) runs the same
script on whatever hardware it finds. These numbers are from a
laptop with an 8 GB NVIDIA RTX PRO 1000 (Blackwell) and an Intel Arc 140T iGPU that borrows system memory.

**1. Detection.** Each plugin ran its own hardware test; all three found devices:

```
plugins:
  loaded   cpu    (pytorches_plugin_cpu.dll)
  loaded   cuda   (pytorches_plugin_cuda.dll)
  loaded   xpu    (pytorches_plugin_xpu.dll)

devices:
  cpu:0    cpu                                      40.6 GiB free / 63.4 GiB  285.4 GFLOP/s
  cuda:0   NVIDIA RTX PRO 1000 Blackwell ...         6.8 GiB free / 8.0 GiB    6.56 TFLOP/s
  xpu:0    Intel(R) Arc(TM) 140T GPU (32GB)         33.5 GiB free / 33.5 GiB  2.36 TFLOP/s
```

**2. The same script on every device** (a 4-layer, 2048-wide MLP, batch 256, one training step):

| device | time per step |
|---|---|
| `cpu:0` (Core Ultra 7 265H, 16 threads) | 33.7 ms |
| `xpu:0` (Intel Arc 140T) | 10.7 ms |
| `cuda:0` (RTX PRO 1000) | 3.6 ms |

**3. Placement.** A model that fits the fast GPU goes there. One that doesn't goes to the
big-memory device, and the planner says why:

```
Placement plan: workload needs ~6.5 GiB

  device   name                               free / total         speed          fits
  cuda:0   NVIDIA RTX PRO 1000 Blackwell Gene 6.8 GiB / 8.0 GiB    6.56 TFLOP/s   no
  xpu:0    Intel(R) Arc(TM) 140T GPU (32GB)   33.5 GiB / 33.5 GiB  2.36 TFLOP/s   yes  <- chosen
  cpu:0    cpu                                40.0 GiB / 63.4 GiB  285.4 GFLOP/s  yes

  -> xpu:0: needs 6.5 GiB; the fastest device (cuda:0) has only 6.8 GiB free, so fall back to xpu:0
```

That run then trains an 805M-parameter, 12-layer MLP on the Arc (about 0.4 s per step), a model that
doesn't fit the RTX's free memory. The speed column comes from a short matmul benchmark run through
each plugin, so it reflects what that plugin can actually do on this machine.
`--dry-run` prints only the placement decisions.

**4. Using both GPUs on one model.** When a model fits no single fast device, the planner can split it
by layers. What to do in that situation depends on the model and the machine (sizes, device speeds, the
cost of moving data between devices), so placement is a set of strategies behind a trait
(`PlacementStrategy`): each proposes a placement with an estimated step time, the fastest wins, and
`plan_model(..., strategy="layer_split")` forces one. Planning needs no tensors; the model is then created
directly on its devices:

```python
plan = pt.plan_model(pt.nn.mlp_graph([8192] * 13, batch=32))   # nothing allocated yet
print(plan)                      # which layers go where, what moves, what it costs, what else was considered
model = pt.nn.mlp([8192] * 13, plan)
```

The 9.1 GiB, 12-layer model above does not fit the RTX's 6.8 GiB. The plan puts layers 0-13 on the RTX
and the rest on the Arc; it trains in about 224 ms per step against 515 ms for the same model entirely on
the Arc, with identical losses (largest relative difference 1.2e-7). See
[`examples/split_model.py`](https://github.com/tonyt123456/pytorches/blob/main/examples/split_model.py).
The step-time estimates run optimistic (127 ms
predicted for that run, 222 ms for the Arc-only run) but rank the options correctly; the Arc's missing
transposed matmul is part of the gap.

## Performance, honestly

Against PyTorch 2.13 on the same laptop (full table and methodology in
[benchmarks/](https://github.com/tonyt123456/pytorches/blob/main/benchmarks/README.md)):

| | matmul 4096² | elementwise add | MLP train step |
|---|---|---|---|
| **CUDA** (RTX PRO 1000) | **1.01x** | **1.01x** | 0.93x |
| **Intel Arc** | 0.50-0.52x | 0.67-0.96x | 0.74-0.80x |
| **CPU** (matmul 1024², add 8M) | 1.16-1.22x | 1.01-1.17x | 0.57-1.66x |

(`PyTorch time / PyTorches time`: above 1.00x we're faster. Ranges are repeated runs on a busy laptop;
the benchmarks README explains why they move.) PyTorches is within noise of PyTorch on CUDA and CPU
throughput, and on Arc matmul it reaches about half of PyTorch's speed. Each gap was closed inside one
plugin, which is the point of the design. The machine is shared: an unusually busy desktop moves every
number here, PyTorch's included.

## Status and roadmap

PyTorches is **early alpha**. It is not a PyTorch replacement yet. The kernels are simple (the
matmul is tiled but untuned, there is no fusion), and the op set is small.

**Working today**

- `f32` tensors; add/sub/mul/div with broadcasting; neg, exp, log, relu, tanh
- sum, mean, 2-D matmul, transpose, reshape; reverse-mode autograd
- **Three backends as plugins:** CPU (reference, bundled), NVIDIA CUDA and Intel GPU (OpenCL), each installed as its own package
- Automatic plugin selection by each plugin's own hardware test
- Device-side tensor creation (constants and a cross-device-reproducible `randn`), in-place updates
- Automatic placement: `pt.plan()` profiles devices and picks one by speed and free memory;
  `pt.place()` builds on it and falls back to the next device on `MemoryError`
- Layer-split placement across devices for models that fit no single one
- Failures are ordinary Python exceptions: `MemoryError`, `ValueError`, `RuntimeError`
- `pt.doctor()` hardware report; minimal `nn` (Linear, ReLU, Sequential, MSE) and SGD
- Differential tests against PyTorch on every detected device
- **PyTorch interop:** DLPack (zero-copy on CPU), safetensors, PyTorch-layout `state_dict`, and reading
  `torch.save` checkpoints without torch

**Known limits**

- A single tensor lives on one device (a model can be split across devices by layer).
- Free memory on the Intel iGPU is an estimate (OpenCL has no free-memory query), and it shares
  system RAM, so a plan can be optimistic when RAM is tight.
- Only `f32`, contiguous tensors, 2-D matmul.

**Next**

- [x] CUDA plugin (NVIDIA, current toolkits; Blackwell via PTX JIT)
- [x] Intel GPU plugin (OpenCL for now; Level Zero / SPIR-V later)
- [x] Memory-aware device choice and layer-split placement
- [x] Separate backend packages with automatic discovery
- [ ] ROCm plugin (AMD) and Metal plugin (Apple)
- [x] DLPack exchange with `torch.Tensor`; `safetensors` and `state_dict` loading
- [ ] Zero-copy DLPack for GPU tensors (needs a pointer-export hook in the plugin ABI)
- [ ] Spill/offload and transfer cost in the planner
- [x] Clean out-of-memory errors (`MemoryError`) and fallback to the next device (`pt.place`)
- [ ] Streams/async in the ABI
- [ ] Strided views, more dtypes (f16/bf16/i64), more ops, a fuller `nn`, more optimizers
- [ ] Fusion and tuned matmul/attention kernels
- [ ] Intel NPU plugin (inference only: static-shape graphs)

**Non-goals (for now):** matching PyTorch's 2,000+ ops, or supporting old GPU generations.
Targeting only current drivers and toolkits is what keeps the project tractable.

---

## How it works

```
            ┌─────────────────────────────────────────────┐
 Python  →  │  pytorches (PyO3)                           │
            ├─────────────────────────────────────────────┤
            │  core: Tensor · autograd · device registry  │   no device code
            ├─────────────────────────────────────────────┤
            │  plugin-abi: stable #[repr(C)] vtable       │   ← the contract
            └───────┬───────────────┬───────────────┬─────┘
                    │ dlopen        │ dlopen        │ dlopen
        ┌───────────▼───┐  ┌────────▼──────┐  ┌─────▼─────────┐
        │ plugin_cpu    │  │ pytorches-cuda│  │ pytorches-xpu │  …
        │ (bundled)     │  │ (own package) │  │ (own package) │
        └───────────────┘  └───────────────┘  └───────────────┘
```

At startup the core finds every installed plugin library, asks each one how many devices it can see, and
registers the ones that see any, as `cuda:0`, `xpu:0` and so on. A plugin that reports zero devices (no
hardware, no driver, wrong generation) is ignored. The plugin owns that hardware test: a CUDA plugin asks
the NVIDIA driver, an Intel plugin asks the OpenCL runtime. Because the core never inspects hardware itself,
supporting a new accelerator never touches the core.

### How plugins are loaded

Plugins are found in the wheel's bundled `plugins/` directory, then in every installed package that declares
a `pytorches.plugins` entry point (that is how `pytorches-cuda` and `pytorches-xpu` are found; the package
is located but its Python code is never imported). Each candidate library passes three gates:

1. **Manifest** (`pytorches_plugin_<name>.json`, optional): its ABI version, OS and CPU architecture are
   checked before the library is loaded, so a plugin that cannot apply to this machine is skipped without
   running any of its code.
2. **Probe**: the library is loaded in a throwaway child process with a timeout, so a driver that hangs or
   crashes while initializing cannot take your program down. A plugin that sees no devices is skipped here.
3. **Load** in your process, which re-checks the ABI and device count.

Probe results are cached in your per-user cache directory (successes for 24 hours, failures for 10 minutes;
a rebuilt library is always probed again). Environment variables:

| Variable | Effect |
|---|---|
| `PYTORCHES_PLUGIN_DIR` | A path list that replaces all plugin discovery (useful when developing a plugin) |
| `PYTORCHES_PROBE=off` | Load plugins directly, skipping the child-process probe |
| `PYTORCHES_PROBE_CACHE=off` | Ignore the probe cache |
| `PYTORCHES_PROBE_TIMEOUT=<seconds>` | Bound the probe (default 20) |
| `PYTORCHES_CACHE_DIR` | Where the probe cache lives |

## Correctness first

Every op is checked against PyTorch itself. [`tests/diff`](https://github.com/tonyt123456/pytorches/tree/main/tests/diff)
runs each op in both libraries on
random inputs (including broadcasting shapes), **on every device it detects**, and compares the
forward values and the gradients. A new backend inherits that oracle for free. The suite also
checks the cross-device RNG, in-place updates, device moves, placement, and that a small training
run reduces its loss.

## For developers

### Building from source

Requires Rust (stable), the MSVC build tools on Windows, and Python 3.11+. Most users should `pip install`
as above instead.

```powershell
git clone https://github.com/tonyt123456/pytorches
cd pytorches

cargo test                                    # core tests (CPU plugin linked statically)
.\scripts\build-plugins.ps1                   # build every plugins/* crate into plugins/bin

python -m venv .venv --system-site-packages
.venv\Scripts\python -m pip install maturin
cd crates\python; ..\..\.venv\Scripts\maturin develop --release; cd ..\..

# GPU backends live in their own repos; install the ones your machine has to run the GPU tests
.venv\Scripts\python -m pip install pytorches-cuda pytorches-xpu

.venv\Scripts\python -m unittest discover -s tests\diff -v
```

Working on a backend? Build it from its repo and `pip install` the wheel, or point `PYTORCHES_PLUGIN_DIR`
at its build output.

| Path | What it is |
|---|---|
| `crates/plugin-abi` | The stable plugin ABI. The only crate a plugin author needs. |
| `crates/core` | `Tensor`, reverse-mode autograd, device registry, plugin loader. |
| `crates/python` | PyO3 bindings plus the Python package (`nn`, `optim`, `doctor`, `plan`), built with maturin. |
| `plugins/cpu` | Reference backend, bundled in the wheel. |
| `plugins/rocm`, `plugins/metal` | Placeholders for AMD and Apple GPUs. They load but report no devices. |
| `tests/diff` | Differential tests against PyTorch. |
| `examples/` | The demo. |

### Writing a plugin

Most users never need this. It's for people adding support for new hardware, or tuning for
existing hardware, without touching the core. The contract between a plugin and the core (who owns what,
required and optional ops, the execution model, errors, versioning, and what "conforming" means) is
specified in [docs/plugin-abi.md](https://github.com/tonyt123456/pytorches/blob/main/docs/plugin-abi.md).

1. `cargo new --lib <name>`, set `crate-type = ["cdylib"]`, depend on
   [`pytorches-plugin-abi`](https://github.com/tonyt123456/pytorches/blob/main/crates/plugin-abi/src/lib.rs).
   The crate is the full contract and is documented inline. Implementing the same struct from C works too.
2. Export `pytorches_plugin_entry() -> *const PluginVTable`, with `abi_version: ABI_VERSION` and
   `struct_size: VTABLE_SIZE` filled in. The vtable covers device enumeration, `alloc`/`free`, host copies,
   `supports_op`, `execute` and `synchronize`. The core does shape inference and allocates output buffers;
   the plugin only fills them. Operands arrive as strided descriptors (broadcasting is stride 0, a transpose is
   a strided copy). `execute` returns a status code; nothing Rust-specific crosses the boundary.
3. Implement the vtable. [`plugins/cpu`](https://github.com/tonyt123456/pytorches/blob/main/plugins/cpu/src/lib.rs)
   is a complete reference in about 270 lines;
   [pytorches-cuda](https://github.com/tonyt123456/pytorches-cuda) and
   [pytorches-xpu](https://github.com/tonyt123456/pytorches-xpu) are full examples of standalone plugin packages.
4. Add a manifest, `pytorches_plugin_<name>.json`, next to the library (see
   [`plugins/cpu/manifest.json`](https://github.com/tonyt123456/pytorches/blob/main/plugins/cpu/manifest.json)).
5. Run the differential tests. If they pass, your backend is correct.

**Shipping a plugin as its own package.** Put the library and its manifest inside a Python package
and declare it with an entry point; `import pytorches` finds it with no configuration:

```toml
[project.entry-points."pytorches.plugins"]
cuda = "pytorches_cuda"   # a module whose directory holds pytorches_plugin_cuda.{dll,so}
```

ABI evolution: ops are append-only. New vtable fields are appended without a version bump; a
plugin built against an older header has a smaller `struct_size` and the core reads only what it
contains. A layout break bumps `ABI_VERSION` (now 4).

## Contributing

The plugin boundary is the main way to contribute: a new backend needs no change to the core.
Issues and PRs are welcome. Run `cargo test` and the differential tests before submitting.

## License

Apache-2.0. See [LICENSE](https://github.com/tonyt123456/pytorches/blob/main/LICENSE).
