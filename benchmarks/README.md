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
| **xpu** (Arc) | matmul 2048² | 2,456-2,479 GFLOP/s | 4,121-4,325 GFLOP/s | 0.57-0.60x |
| | matmul 4096² | 2,133-2,330 GFLOP/s | 4,418-4,464 GFLOP/s | 0.48-0.52x |
| | add, 64M floats | 59-87 GB/s | 86-91 GB/s | 0.67-0.96x (see pitfall 4) |
| | sum, 64M floats | 82-87 GB/s | 86-87 GB/s | 0.95-1.01x |
| | MLP train step | 96-98 steps/s | 119-132 steps/s | 0.74-0.80x |
| **cpu** (16 threads) | matmul 512² | 541-805 GFLOP/s | 468-874 GFLOP/s | 0.92-1.72x |
| | matmul 1024² | 693-763 GFLOP/s | 579-637 GFLOP/s | 1.16-1.22x |
| | add, 8M floats | 67-78 GB/s | 64-72 GB/s | 1.01-1.17x |
| | sum, 8M floats | 96-211 GB/s | 82-111 GB/s | 1.15-1.91x |
| | MLP train step (4×512, batch 64) | 544-1,218 steps/s | 706-960 steps/s | 0.57-1.66x |

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
- **Arc: elementwise ops and `sum` are at parity, matmul is at ~0.5-0.6x, and a training step is at
  0.74-0.80x** (it started the day at 0.19x, 0.23x and 0.26x for add, matmul and the step). What did it:
  - A caching allocator (every `a + b` used to create and commit a fresh 256 MB buffer): `add` 17 -> 80+ GB/s.
  - A sub-group matmul kernel (N % 32, K % 16; whole 16-row blocks, the leftover rows use the general
    kernel): 1.1 -> 2.3-2.5 TFLOP/s, about 2x. Other shapes use the older general kernel.
  - A fused `AXPY` optimizer update: the SGD step 4.2 -> 1.8 ms.
  - A tiled local-memory transpose for `COPY`s that are 2-D transposes (what backward produces):
    42 -> 61-64 GB/s.
  - 4-wide loads in the row `sum`: 68 -> 82-87 GB/s.
- **The Arc matmul gap is vector-unit efficiency, not matrix engines.** fp32 matmul in PyTorch runs on the
  ordinary vector units here: a pure-FMA kernel measures a **4.1 TFLOP/s** fp32 peak on this Arc 140T
  (128 EUs), and PyTorch reaches 4.0-4.3 of it. Intel's matrix instructions (`DPAS`, advertised through
  `cl_intel_subgroup_matrix_multiply_accumulate`) take fp16/bf16/int8, not fp32, so they matter only for a
  reduced-precision path. Our kernel is at ~58% of the vector peak. Tried, and none beat ~2.4-2.5 TFLOP/s:
  wider or narrower register tiles, sharing B across a work-group, tile-order swizzles, register
  prefetch (spills the 128-register file), hardware `prefetch()`, the 256-GRF mode (halves occupancy) and
  32-bit index math. Ruled out as the cause by measurement: spills (0 bytes), cache aliasing (non-power-of-two
  strides behave the same), cache capacity (a cache-resident B is no faster), FMA issue and broadcasts
  (removing half the FMAs left the time unchanged). That leaves the load path itself; what is still untried
  is local-memory staging and Intel's 2-D block loads, which this driver does not expose
  (`cl_intel_subgroup_2d_block_io` is not advertised). A transposed-B kernel (each lane loading its own
  column's k-values) was correct but 4x slower (0.6 TFLOP/s), so backward still materializes transposes on
  Arc, now through the faster transpose kernel.
- **CPU is at parity with MKL** (it started 10-50x behind: single-threaded scalar kernels, a fresh
  allocation per op). What changed, in the order it paid off:
  - Threads plus AVX2/FMA inner loops for elementwise ops, in place for broadcasts (the old code copied
    every strided operand first), and vector `exp`/`log`/`tanh` (libm has no vector form, so these were
    scalar calls): `add` 8.5 -> 67-78 GB/s, `sum` 1.8 -> 96-211 GB/s.
  - A packed, cache-blocked GEMM with a 6x16 AVX2/FMA microkernel: 24 -> 693-763 GFLOP/s at 1024².
    Packing reads arbitrary strides, so transposed operands (`MATMUL_T`) cost nothing, and `AXPY`
    fuses the optimizer update.
  - A caching allocator (capped at an eighth of RAM) and a custom thread pool that spins briefly before
    sleeping. Dispatching to a sleeping rayon worker costs 53-119 us on Windows, longer than a small
    matmul; with a 100 us spin it costs 2-10 us. The MLP step went 53 -> ~1,000+ steps/s.
  - The spin window is a real trade-off, measured: none loses the small-op speed (MLP step 520-700 vs
    1,060-1,180 steps/s), and 300 us or more makes large ops slower (matmul 1024: ~720 -> 380-520
    GFLOP/s, add 78 -> 46 GB/s), likely because workers parked in a spin loop get moved to slow cores
    on this hybrid CPU. `PYTORCHES_CPU_SPIN_US` and `PYTORCHES_CPU_THREADS` expose both knobs.
  - Results do not depend on the thread count: work is cut into fixed chunks and partial sums are
    combined in a fixed order, so runs are bit-identical.
  - Where it is still behind: matmuls of 128-384 (a fraction of a millisecond) still pay some dispatch
    cost; and under heavy background load a spinning pool is more fragile than MKL (a worker descheduled
    mid-task stalls the job until it runs again).

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
5. **A busy desktop moves CPU numbers a lot, and moves them for PyTorch too.** While this was measured,
   Task Manager, Slack, Zoom and a browser kept the machine at 20-45% CPU even when idle; MKL's matmul
   readings ranged 19-874 GFLOP/s across runs (19 at 2048², in one run), and ours showed the same kind of
   swing. Fork-join libraries stall when a worker is descheduled mid-task. Compare medians of repeated
   runs, close other applications for a number you intend to keep, and treat the ranges in the table as
   the result rather than any single run.
