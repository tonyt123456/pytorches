# PyTorches TODO

Priorities: **P0** blocks the next milestone · **P1** needed for a usable v0.1 · **P2** important, not urgent · **P3** later / nice to have.
Each phase ends with an exit test, which is the thing that must be true to call the phase done.

Legend: `[x]` done · `[ ]` open

---

## Phase 0: Foundation (done)

- [x] Cargo workspace, Rust 2024, MSVC toolchain on Windows
- [x] `Tensor` with `f32` storage, reverse-mode autograd (accumulation, reuse, broadcast reduction)
- [x] Ops: add/sub/mul/div (broadcasting), neg/exp/log/relu/tanh, sum/mean, 2-D matmul/transpose, reshape
- [x] Stable `#[repr(C)]` plugin ABI (`crates/plugin-abi`), version-checked at load
- [x] Runtime plugin loader and device registry (`$PYTORCHES_PLUGIN_DIR`, `./plugins/bin`)
- [x] CPU reference plugin as a real shared library
- [x] Python bindings (PyO3 + maturin), device/plugin API, differentiable `.to()`
- [x] Differential tests vs PyTorch 2.13 (forward + gradients)
- [x] README, git repo, pushed to GitHub

---

## Phase 1: Hardening the foundation (before adding GPUs)

Fix the cracks now, while there is one backend. These get more expensive with every plugin.

**ABI and plugin contract (P0, do before writing the CUDA plugin)**
- [ ] **Streams/async execution in the ABI.** Add a stream/queue handle (`create_stream`, `destroy_stream`, stream arg on `execute`/copies, event record/wait). GPUs are async, and a sync-only ABI would cripple them. Bump `ABI_VERSION` to 2 while there is only one plugin.
- [ ] **Async copies + pinned host memory** (`alloc_host`, `copy_*_async`) for fast host↔device transfers
- [x] **Device-to-device copy entry point** for same-plugin copies (skip the host staging in `Tensor::to`)
- [ ] **Capability query**: dtypes supported, max alloc, memory-pool support, so the planner can reason about devices
- [ ] **Error model**: per-call error strings (thread-local), consistent status codes, documented ownership rules
- [ ] **Thread-safety contract** written down and tested (concurrent `execute` from multiple threads)
- [ ] ABI conformance test suite: a plugin-agnostic Rust/Python harness any plugin can run (alloc/copy round-trip, every advertised op vs CPU)
- [ ] C header generated from `plugin-abi` (cbindgen) so C/C++ plugin authors don't hand-copy structs

**Core correctness and robustness**
- [ ] **Replace panics with `Result`** in core (shape mismatch, device mismatch, unsupported op) and raise proper Python exceptions (`ValueError`, `RuntimeError`)
- [ ] `no_grad` context and `Tensor.requires_grad_()` semantics matching PyTorch
- [ ] In-place update path for parameters (needed by optimizers; today the README example re-wraps leaves)
- [ ] Gradient of a non-scalar via explicit `grad_output` (`backward(gradient=...)`)
- [ ] Retain-graph semantics: free graph after backward unless asked, avoid leaks
- [ ] Audit `Arc` reference cycles in autograd closures (output captured by backward closures)
- [ ] Zero-element and 0-dim tensor edge cases tested

**Python packaging and discovery**
- [x] Locate `plugins/` relative to the installed package instead of CWD / env var
- [ ] `pip install`-able wheel (maturin build) that bundles the CPU plugin
- [ ] Silence the pyo3 deprecation warning at build time
- [ ] Type stubs (`.pyi`) and docstrings for the Python API

**Repo hygiene**
- [ ] `.gitattributes` to pin LF line endings (CRLF warnings on Windows)
- [ ] **CI** (GitHub Actions): `cargo fmt --check`, `clippy -D warnings`, `cargo test`, build CPU plugin, run Python diff tests (CPU-only runners)
- [ ] `rustfmt.toml` / `clippy` config; `CONTRIBUTING.md`; issue and PR templates
- [ ] Pin PyTorch version for the oracle in CI (`requirements-test.txt`)

**Exit test:** a toy second plugin written from *only* the ABI header (C or Rust, e.g. a "slow-cpu" plugin) loads, passes the conformance suite, and runs the differential tests with zero core changes.

---

## Phase 2: Interop with PyTorch

- [ ] **DLPack** export/import (`__dlpack__`, `__dlpack_device__`, `pt.from_dlpack`): zero-copy with `torch.Tensor` on CPU (P0)
- [ ] DLPack for CUDA / XPU device memory (needs plugin hooks to export device pointers and sync)
- [ ] `safetensors` read/write
- [ ] Load PyTorch `.pt` `state_dict`s via a restricted unpickler (no arbitrary code execution)
- [ ] Parameter/module naming identical to PyTorch so weights load with no remapping
- [ ] `pt.from_torch(t)` / `t.to_torch()` convenience helpers
- [ ] Import `torch.export` / FX graphs or ONNX → PyTorches ops (run existing models without rewriting)
- [ ] **Compatibility target:** `import pytorches as torch` works for a defined subset. Publish the exact list of supported APIs and keep it in `docs/compat.md`
- [ ] Differential-test harness extended to modules and full models (MLP, small CNN, tiny transformer)
- [ ] Optional: register PyTorches as a `torch.compile` backend or a PyTorch `PrivateUse1` device

**Exit test:** load a real PyTorch checkpoint (a small MLP and a small transformer), run inference in PyTorches, and match PyTorch logits within tolerance; round-trip tensors through DLPack with no copy (verified by pointer equality).

---

## Phase 3: CUDA plugin (NVIDIA RTX PRO 1000 Blackwell, sm_120, CUDA 13.x)

- [x] `plugins/cuda` crate; decide the binding strategy (`cudarc`-style driver API wrapper vs raw `cuda-sys`), prefer the **driver API + PTX/cubin loading** so the plugin doesn't hard-link a specific CUDA runtime
- [x] Device enumeration, `device_info` with real free/total memory (`cuMemGetInfo`)
- [x] Allocator: caching/pooled allocator to avoid `cudaMalloc` per tensor
- [ ] Streams and events wired to the Phase 1 ABI
- [ ] Host↔device copies (pinned + async)
- [x] Elementwise kernels with strided/broadcast operands (NVRTC or build-time PTX; **include sm_120 PTX for JIT forward-compat**)
- [x] Reductions (`sum_axis`)
- [ ] Matmul via cuBLAS/cuBLASLt (dynamic load), with a plain fallback kernel
- [x] Graceful behavior when no NVIDIA GPU/driver is present (0 devices, no load error spam)
- [x] Runs the full ABI conformance suite and differential tests (CPU oracle; installed PyTorch is the XPU build, so compare CUDA results against CPU)
- [ ] Benchmarks vs the CPU plugin and vs PyTorch-CUDA if available (matmul, elementwise, softmax-sized reductions)
- [x] Handle 8 GB VRAM limits cleanly: OOM returns a status, never aborts

**Exit test:** the diff suite passes with `device="cuda:0"`; matmul 4096² is faster than the CPU plugin by a large margin; no leaked allocations after a 1000-iteration training loop.

---

## Phase 4: Intel Arc plugin (Arc 140T, ~47 GB shared memory)

- [x] `plugins/xpu` crate; choose the runtime (Level Zero direct vs SYCL/oneAPI), prefer **Level Zero + SPIR-V** to avoid shipping the full oneAPI toolchain
- [x] Device enumeration and memory reporting for iGPU shared memory
- [ ] Allocator suited to unified/shared memory (zero-copy host access where possible)
- [x] Kernel generation: SPIR-V for elementwise/reduction ops (hand-written first; shared compiler later, see Phase 6)
- [ ] Matmul via oneMKL/oneDNN (dynamic load) with a fallback kernel
- [x] Conformance + differential tests; compare against PyTorch XPU (installed) as a second oracle
- [ ] Benchmarks: where does Arc beat CUDA by virtue of memory capacity (large tensors that do not fit 8 GB)?

**Exit test:** diff suite passes on `xpu:0`; a tensor workload larger than the RTX's 8 GB runs on Arc.

---

## Phase 5: Heterogeneous execution (the differentiator)

- [x] Device profiling: measured bandwidth, matmul throughput, and transfer cost between every device pair, cached per machine (`pt.calibrate()`)
- [x] Multi-device `Tensor` placement API (`device="auto"`)
- [x] **Planner:** given a graph and device memory/speed profiles, choose placement (fits → fastest device; doesn't fit → bigger device; or split by layer)
- [ ] **Spill/offload:** when a device nears OOM, evict cold tensors to host or another device instead of failing
- [ ] Pipeline/layer offload for models larger than the fast device (hot layers on CUDA, rest on Arc)
- [ ] Overlap transfers with compute using streams/events
- [ ] Peer-to-peer or shared-memory fast paths where hardware allows
- [x] Visualization/debug: `pt.explain_plan(model)` shows what ran where and why, plus transfer cost
- [ ] Failure handling: device disappears or driver resets mid-run

**Exit test:** run a model that doesn't fit in 8 GB using both GPUs together, faster than running it entirely on Arc, with results matching a single-device run.

---

## Phase 6: Performance and compiler

- [ ] Strided views (transpose/slice/expand without copies), `contiguous()`, `view` vs `reshape`
- [ ] More dtypes: f16, bf16, i32, i64, bool (ABI additions)
- [ ] Op fusion: elementwise chains into one kernel
- [ ] Shared kernel IR/codegen: one op description lowered to PTX (CUDA), SPIR-V (Arc), AMDGPU (ROCm), instead of hand-writing each kernel per plugin
- [ ] Tuned matmul, attention (FlashAttention-style), softmax, layernorm, conv kernels
- [ ] CPU plugin: SIMD (AVX2/AVX-512/NEON), threading (rayon), blocked matmul
- [ ] Memory planner for training (activation lifetimes, optional recompute/checkpointing)
- [ ] Continuous benchmark tracking in CI (regression alerts)

---

## Phase 7: Library surface (usable for real models)

- [ ] Ops: softmax, log_softmax, cross-entropy, layernorm, dropout, embedding, conv1d/2d, pooling, batched matmul, indexing/slicing, cat/stack, where, comparisons, argmax, pow/sqrt/abs/clamp, sigmoid/gelu/silu
- [ ] N-d `matmul` with broadcasting
- [ ] `nn.Module`, `Parameter`, `Linear`, `Sequential`, activations, loss modules
- [ ] Optimizers: SGD (+momentum), Adam/AdamW; LR schedulers
- [ ] `Dataset`/`DataLoader` basics
- [ ] Random: seedable generators with cross-device determinism where feasible
- [ ] Save/load checkpoints
- [ ] Mixed precision (autocast-style) and gradient scaling
- [ ] Higher-order gradients (`create_graph`) and `torch.autograd.grad`-style API
- [ ] Custom op registration from Python/Rust and from plugins

**Exit test:** train MNIST MLP and a small CNN to PyTorch-comparable accuracy on CPU, CUDA and Arc; fine-tune/infer a small transformer.

---

## Phase 8: More backends and distribution

- [ ] ROCm plugin (current ROCm/HIP, gfx942/950, RDNA3/4)
- [ ] Vulkan/SPIR-V fallback plugin for "any GPU"
- [ ] Apple Metal plugin
- [ ] Google TPU via StableHLO/XLA plugin
- [ ] Intel NPU plugin (inference only, static shapes, OpenVINO / Level Zero NPU)
- [ ] **Hardware detection** (`pt.doctor()`): report GPUs, drivers, which plugins match
- [ ] **On-demand plugin download:** signed manifest mapping (OS, arch, driver, compute capability) → plugin build; hash-pinned cache in `~/.cache/pytorches/plugins`
- [ ] Offline/air-gapped mode and lockfile recording the resolved plugin set (reproducibility)
- [ ] Plugin signing and trust policy (loading native code is a security boundary)
- [ ] Multiple plugin versions side by side; driver-upgrade re-resolution
- [ ] Linux and macOS support in CI and build scripts (the current script is PowerShell/Windows)

---

## Phase 9: Project and community

- [ ] `docs/` site: architecture, plugin author guide, ABI reference, compatibility matrix
- [ ] `docs/plugin-authoring.md` with a C example and a Rust example, plus a plugin template repo (`cargo generate`)
- [ ] Versioning and release process; semantic versioning for core vs ABI
- [ ] Publish crates (`pytorches-plugin-abi`) and wheels; reserve the PyPI name
- [ ] Security policy (`SECURITY.md`), code of conduct, governance for third-party plugins
- [ ] Benchmarks page with honest numbers (including where we lose to PyTorch)
- [ ] Example gallery: linear regression, MNIST, multi-GPU placement demo

---

## Open design questions

- **Kernel strategy:** hand-written kernels per plugin, or a shared compiler (MLIR / Triton-style) feeding all plugins? Decide before Phase 6; it changes what a "plugin" has to implement.
- **ABI granularity:** keep per-op `execute`, or add a "run this fused graph" entry point so plugins can compile whole subgraphs?
- **Where do planner and cost model live?** Core, or a separate pluggable crate?
- **Ownership of device pointers across plugins** (for DLPack and cross-plugin zero-copy): who frees, and how do events synchronize?
- **Autograd ownership:** keep tape-based Rust autograd, or lower training to a compiled backward graph?
- **Licensing for third-party plugins** (closed-source vendor plugins allowed? ABI is the boundary).

## Known limitations (today, earlier list)

- Cross-device copies always stage through host memory.
- Panics, not `Result`s, on shape/device errors (surface as `PanicException` in Python).
- Only `f32`, contiguous tensors; transposes and broadcasts are materialized.
- Plugin discovery depends on env var / CWD, not install location.
- Windows-only build script; no CI yet.
- CPU plugin is naive (single-threaded, no SIMD).

---

## Added after the first multi-GPU demo (2026-10-06)

Found while building and testing the CUDA and Intel Arc plugins and the demo.

**P0**
- [ ] **Out-of-memory must be an error, not a panic.** Plugins return a clean `STATUS_OUT_OF_MEMORY`, but the core asserts on any non-OK status, so the Python call dies with a `PanicException` (a `BaseException`). Make `Buffer::alloc`/`run_op` return `Result`, surface `MemoryError` in Python, and let the planner retry on another device.
- [ ] **Planner must account for shared memory.** The Arc iGPU draws from system RAM, and OpenCL has no free-memory query, so its "free" figure is `total - tracked`. Combine it with host free memory (and warn when a plan could page).
- [ ] Planner decisions need a stable benchmark: calibration now warms up and takes the best of three, but add a variance check and an optional on-disk cache keyed by device + driver version.

**P1**
- [ ] ABI: byte offsets (or sub-range handles) on `copy_*` so sub-range reads don't need a strided COPY
- [ ] ABI: stream/event handles (the CUDA caching allocator currently relies on a single stream; `alloc`/`free` will need a stream or fence argument)
- [ ] Core: use `copy_device_to_device` for same-device moves; peer copies where plugins support them
- [ ] CUDA: tune matmul (cuBLAS/cuBLASLt via dynamic load, or a better kernel); measured ~4.8 TFLOP/s at 4096³
- [ ] Arc: tune matmul (sub-groups, DPAS/matrix extensions); measured ~1.0 TFLOP/s, and exp is compute-bound at 14-19 GB/s
- [ ] Arc: move from OpenCL to Level Zero + SPIR-V (OpenCL is a pragmatic first runtime)
- [ ] Matmul and `sum_axis` accept only contiguous inputs in the GPU plugins; the core always passes contiguous ones, but fix it before strided views land
- [ ] Test the no-hardware path on a machine without an NVIDIA GPU / Intel GPU (plugins must return 0 devices quietly)
- [ ] Multi-thread stress tests for plugin entry points (only per-device mutexes verified so far)
- [ ] `examples/demo.py`: record a short screen capture and add it to the README
- [ ] CI: runners have no GPU, so run CPU tests there and keep GPU conformance as a documented local step; keep heavy allocation tests opt-in (`--ignored`)

**P2**
- [ ] Arc plugin env overrides (`PYTORCHES_XPU_VERBOSE`, `_ALLOC=host`, `_BUILD_OPTS`) documented in the plugin guide
- [ ] Planner: model-aware estimates (today `estimate_mlp_training_bytes` is MLP-only)
- [ ] `pt.doctor()`: report driver versions and why a plugin was skipped in more detail
