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

- [x] **DLPack** export/import (`__dlpack__`, `__dlpack_device__`, `pt.from_dlpack`): zero-copy with `torch.Tensor` on CPU (P0)
- [ ] DLPack for CUDA / XPU device memory (needs plugin hooks to export device pointers and sync). Today GPU tensors export through `.to("cpu:0")`, and CPU/XPU producers import via a host copy
- [x] `safetensors` read/write
- [x] Load PyTorch `.pt` `state_dict`s via a restricted unpickler (no arbitrary code execution)
- [x] Parameter/module naming identical to PyTorch so weights load with no remapping
- [x] `pt.from_torch(t)` / `t.to_torch()` convenience helpers
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
- [x] **Out-of-memory is an error, not a panic.** The core raises a typed `Error` (`OutOfMemory` / `Invalid` / `Backend`), `try_run` catches it, and Python gets `MemoryError` / `ValueError` / `RuntimeError`. `pt.place()` retries on the next device. Remaining: make the core API return `Result` natively instead of unwinding with typed payloads.
- [ ] **Planner must account for shared memory.** The Arc iGPU draws from system RAM, and OpenCL has no free-memory query, so its "free" figure is `total - tracked`. Combine it with host free memory (and warn when a plan could page).
- [ ] Planner decisions need a stable benchmark: calibration now warms up and takes the best of three, but add a variance check and an optional on-disk cache keyed by device + driver version.

**P1**
- [ ] ABI: byte offsets (or sub-range handles) on `copy_*` so sub-range reads don't need a strided COPY
- [ ] ABI: stream/event handles (the CUDA caching allocator currently relies on a single stream; `alloc`/`free` will need a stream or fence argument)
- [ ] Core: use `copy_device_to_device` for same-device moves; peer copies where plugins support them
- [ ] CUDA: tune matmul (cuBLAS/cuBLASLt via dynamic load, or a better kernel); measured ~4.8 TFLOP/s at 4096³
- [ ] Arc: matmul is at 2.0-2.4 TFLOP/s (sub-group kernel, up from 1.0) against a measured 4.1 fp32 peak and PyTorch's 4.0-4.3. The matrix engines (DPAS) are fp16/bf16/int8 only, so they are for a reduced-precision path, not this gap. Exp is compute-bound at 14-19 GB/s.
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

## Performance gaps measured against PyTorch 2.13 (2026-10-06; see benchmarks/README.md)

Ordered by payoff.

- [ ] **CPU plugin: threading + SIMD + blocked matmul.** 10-50x behind MKL (matmul 22 vs 611 GFLOP/s, sum 1.8 vs 88 GB/s). Start with rayon over rows/chunks and a packed, vectorized matmul.
- [x] **Intel plugin: caching allocator.** `add` on 64M floats 17 -> 80 GB/s (PyTorch 86). Freed buffers are reused by size bucket; the cache is flushed when an allocation fails, and cached bytes count as free in `device_info`.
- [~] **Intel plugin: matmul.** Sub-group kernel for N % 32, K % 16 (any M; about 2x, 2.4 vs PyTorch's 4.1-4.4 TFLOP/s; see benchmarks/README.md for what was tried). Open: local-memory staging or 2-D block loads (need a driver that exposes `cl_intel_subgroup_2d_block_io`) for the rest of the gap, a fast path for N or K that are not multiples of 32/16, and a DPAS fp16/bf16 path (needs those dtypes).
- [x] Intel plugin: `sum` (4-wide loads) at parity; tiled 2-D transpose for COPY (61-64 GB/s, was 42).
- [ ] Intel plugin: `MATMUL_T` is not implemented (core materializes transposes for backward). A direct transposed-B kernel was 4x slower than the plain kernel plus a copy, so it needs a different design. CUDA's transpose copy (148 GB/s of ~267) could use the same tiled kernel.
- [x] **CUDA plugin: matmul** now calls cuBLAS (runtime-loaded, built-in kernel as fallback): 1.00x of PyTorch fp32. Still open: tune the fallback kernel (4-5 vs 6.5 TFLOP/s), and an opt-in TF32 / bf16 tensor-core path (PyTorch reaches ~8.5 TFLOP/s with TF32; needs the f16/bf16 dtypes).
- [x] CUDA `sum` / `add`: vectorized (float4) kernels; at parity. The earlier gap was GPU memory throttling during measurement (benchmarks/README.md, pitfall 4).
- [x] MLP step on CUDA: 0.54x -> ~0.93x via `MATMUL_T` (no transposed copies), skipping unneeded input gradients, and a fused `AXPY` optimizer update. Remaining ~7%: per-op allocate/launch overhead (no fusion, no CUDA graphs).
- [ ] Optional ABI ops `MATMUL_T` and `AXPY`: CUDA has both, Arc has `AXPY`, the CPU plugin has neither (core falls back to the old path). Add `MATMUL_T` to Arc and both to CPU.
- [x] `calibrate()` bounds its queued work (syncs every 4 matmuls). It used to enqueue for a fixed amount of host time, which was harmless while Arc allocation was slow but queued minutes of GPU work once allocation was cached. The differential suite went from ~20 s to ~2.4 s.
- [ ] Benchmarks: add a regression guard (track numbers per commit), and extend workloads (attention-shaped matmuls, softmax, layernorm) once those ops exist.
