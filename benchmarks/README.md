# Benchmarks: PyTorches vs PyTorch

`compare.py` runs identical workloads in both libraries on every device they share and prints a table.
`bench.py` is the single-process worker; each (library, device) pair runs in its own process so GPU
memory, kernel JIT and caches never leak between measurements.

```powershell
# PyTorches + a PyTorch build per device (a CUDA build for cuda, an XPU build for xpu)
python benchmarks\compare.py `
    --pytorches-python .venv\Scripts\python.exe `
    --torch-python     .venv\Scripts\python.exe `
    --torch-python     cuda=.venv-torch-cuda\Scripts\python.exe
```

Setting up the CUDA comparison (a separate venv, about 3 GB, leaves your main environment alone):

```powershell
python -m venv .venv-torch-cuda
.venv-torch-cuda\Scripts\python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu132
```

## Results (one laptop, 2026-10-06)

NVIDIA RTX PRO 1000 Blackwell (8 GB, sm_120), Intel Arc 140T iGPU, 63 GB RAM, Windows 11.
PyTorch 2.13.0 (`+cu132` for CUDA, `+xpu` for Arc and CPU). PyTorches at commit time of this file.
Float32 throughout; PyTorch's default float32 matmul precision ("highest", no TF32), so both sides do
true fp32. Median of repeated runs after a time-based warm-up. The last column is
`PyTorch time / PyTorches time`: above 1.00x PyTorches is faster, below it PyTorch is.

| device | workload | PyTorches | PyTorch | PyTorches speed |
|---|---|---|---|---|
| **cuda** | matmul 2048² | 6,774 GFLOP/s | 6,678 GFLOP/s | **1.01x** |
| | matmul 4096² | 6,478 GFLOP/s | 6,438 GFLOP/s | **1.01x** |
| | add, 64M floats | 269-341 GB/s | 337 GB/s | 0.8-1.01x (see pitfall 4) |
| | sum, 64M floats | 259-319 GB/s | 269-274 GB/s | 0.95-1.19x (see pitfall 4) |
| | MLP train step (4×2048, batch 256) | 261-267 steps/s | 283 steps/s | 0.91-0.94x |
| **xpu** (Arc) | matmul 2048² | 2,158 GFLOP/s | 4,171 GFLOP/s | 0.52x |
| | matmul 4096² | 2,028 GFLOP/s | 4,057 GFLOP/s | 0.50x |
| | add, 64M floats | 80 GB/s | 86 GB/s | 0.93x |
| | sum, 64M floats | 63 GB/s | 82 GB/s | 0.77x |
| | MLP train step | 70 steps/s | 116 steps/s | 0.60x |
| **cpu** | matmul 1024² | 22 GFLOP/s | 611 GFLOP/s | 0.04x |
| | add, 8M floats | 8.5 GB/s | 64 GB/s | 0.13x |
| | sum, 8M floats | 1.8 GB/s | 88 GB/s | 0.02x |
| | MLP train step (4×512, batch 64) | 49 steps/s | 443 steps/s | 0.11x |

(Workload sizes are smaller on CPU, identical for both libraries on the same device.)

## What this says

- **PyTorch is faster almost everywhere.** It has years of tuning behind it; PyTorches is early alpha.
- **CUDA is at parity on matmul and memory-bound ops, and within ~7% on a training step.**
- **CUDA matmul calls cuBLAS**, loaded at runtime (the plugin links nothing at build time), so it matches
  PyTorch. If cuBLAS is not found the plugin falls back to its own 128x128 tiled kernel, which reaches
  about 4-5 TFLOP/s here (0.67-0.75x). `PYTORCHES_CUBLAS=0` forces the fallback; a path loads that library.
  PyTorch can go further (~8.5 TFLOP/s on this GPU) by enabling TF32 tensor cores, which we don't use.
- **The MLP step went from 0.54x to ~0.93x** without touching matmul again: backward uses transposed
  operands through cuBLAS flags (`MATMUL_T`) instead of materializing copies, skips the gradient of inputs
  that don't need one (the first layer's data), and the SGD update is one fused `AXPY` pass per tensor
  instead of a multiply, a subtract and a copy. Profile of one step before/after: forward 1.2 ms,
  backward 2.95 -> 2.1 ms, optimizer 1.26 -> 0.45 ms. The remaining gap is per-op launch overhead.
- **Arc: elementwise ops are at parity, matmul is at ~0.5x, and a training step is at 0.60x** (it started
  the day at 0.19x, 0.23x and 0.26x). Three changes did it:
  - A caching allocator (every `a + b` used to create and commit a fresh 256 MB buffer): `add` 17 -> 80 GB/s.
  - A sub-group matmul kernel for tile-friendly shapes (M % 16, N % 32, K % 16): 1.1 -> 2.2 TFLOP/s, about 2x.
    Other shapes use the older general kernel.
  - A fused `AXPY` optimizer update: the SGD step 4.2 -> 1.9 ms.
- **The Arc matmul gap is vector-unit efficiency, not matrix engines.** fp32 matmul in PyTorch runs on the
  ordinary vector units here: a pure-FMA kernel measures a **4.1 TFLOP/s** fp32 peak on this Arc 140T
  (128 EUs), and PyTorch reaches 4.0-4.3 of it. Intel's matrix instructions (`DPAS`, advertised through
  `cl_intel_subgroup_matrix_multiply_accumulate`) take fp16/bf16/int8, not fp32, so they matter only for a
  reduced-precision path. Our kernel is at ~53% of the vector peak. Tried and not better, with spills ruled
  out as the cause: wider or narrower register tiles, sharing B across a work-group, tile-order swizzles,
  explicit load prefetch (spills the 128-register file) and the 256-GRF mode (halves occupancy). The next step
  is likely SLM-staged tiles or a different load pattern, and measuring with a standalone harness first.
- **CPU is 10-50x behind**: single-threaded scalar kernels against multi-threaded MKL.

The CPU and Arc gaps are known, simple-to-explain, and the top items on the performance roadmap.
The point of PyTorches' architecture is that each of them is fixable inside one plugin.

## Measurement pitfalls found along the way

Written down because they produced a wrong headline once:

1. **Laptop GPUs idle at a low clock** (this one: 960 MHz idle vs 3,090 MHz max) and need about a second of
   sustained load to ramp up. A fixed number of warm-up calls let one library be timed while the GPU was
   still ramping and the other at full clocks, which briefly showed PyTorches *beating* cuBLAS by 1.2-1.4x.
   `bench.py` now warms up for at least 1.5 s of wall time on GPUs before timing anything.
2. **Don't trust a surprising win without checking correctness.** Running the same matrices through both
   libraries gave a max difference of exactly 0 against cuBLAS, versus 5.8e-4 against a CPU matmul (a
   different summation order), which shows both that our kernel is right and that the check can fail.
3. GPU numbers still vary by a few percent run to run (power state, thermals, other load). Treat
   differences under ~10% as noise.
4. **Memory-bound numbers depend on the GPU's power state.** On this laptop the NVIDIA card is under a
   software power cap (`nvidia-smi` reports throttle reason `0x4`), and its memory clock steps between
   11,801 MHz (~335 GB/s on `add`) and 9,001 MHz (~267 GB/s). The high state is a short burst, seen mostly
   after the GPU has been idle; sustained load settles at the low one. PyTorch behaves identically (330 ->
   263 GB/s in a one-second-window test), so a single timing can show either library "20% slower" depending
   only on which state it landed in. Both kernels reach the same figure in either state, which is why the
   add/sum rows above are ranges. Compare sustained numbers (~267 GB/s here), and log the memory clock
   (`nvidia-smi --query-gpu=clocks.mem,clocks_throttle_reasons.active`) with any result you keep.
   Running a heavy Intel-GPU load alongside did not change the NVIDIA state measurably (5 of 5 runs
   throttled, against 4 of 5 without it), so the Arc's load is not the explanation by itself, though the
   power budget is shared across the laptop and that was not ruled out. The desktop (`dwm`, etc.) keeps the
   Intel GPU at ~15-25% regardless, and our CUDA benchmark does not touch it.
