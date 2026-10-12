# Benchmarks: PyTorches vs PyTorch

`compare.py` runs identical workloads in both libraries on every device they share and prints a table.
`ops_compare.py` times individual ops (broadcast add, relu, exp, transpose copy, reductions, matmul
variants) against PyTorch on one device to find the outliers worth fixing. `bench.py` is the single-process worker; each (library, device) pair runs in its own process so GPU
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

## Results (one laptop; CPU rows 2026-10-06, CUDA and Arc rows re-measured 2026-10-12)

NVIDIA RTX PRO 1000 Blackwell (8 GB, sm_120), Intel Arc 140T iGPU, 63 GB RAM, Windows 11.
PyTorch 2.13.0 (`+cu132` for CUDA, `+xpu` for Arc and CPU). PyTorches at commit time of this file.
Float32 throughout; PyTorch's default float32 matmul precision ("highest", no TF32), so both sides do
true fp32. Median of repeated runs after a time-based warm-up. The last column is
`PyTorch time / PyTorches time`: above 1.00x PyTorches is faster, below it PyTorch is.

| device | workload | PyTorches | PyTorch | PyTorches speed |
|---|---|---|---|---|
| **cuda** | matmul 2048² | 7,043-7,107 GFLOP/s | 6,975-7,146 GFLOP/s | 0.99-1.02x |
| | matmul 4096² | 6,898-7,110 GFLOP/s | 6,269-7,084 GFLOP/s | 0.98-1.13x |
| | add, 64M floats | 266-340 GB/s | 334-338 GB/s | 0.79-1.01x (see pitfall 4) |
| | sum, 64M floats | 261-324 GB/s | 272-315 GB/s | 0.83-1.19x (see pitfall 4) |
| | MLP train step (4×2048, batch 256) | 264-281 steps/s | 289-296 steps/s | 0.91-0.96x |
| **xpu** (Arc) | matmul 2048² | 3,208-3,447 GFLOP/s | 2,948-4,314 GFLOP/s | 0.80-1.12x (see pitfall 6) |
| | matmul 4096² | 2,958-3,315 GFLOP/s | 3,253-4,442 GFLOP/s | 0.75-0.94x (see pitfall 6) |
| | add, 64M floats | 79-86 GB/s | 76-87 GB/s | 0.99-1.05x |
| | sum, 64M floats | 82-84 GB/s | 64-86 GB/s | 0.95-1.29x |
| | MLP train step | 107-111 steps/s | 123-128 steps/s | 0.84-0.87x |
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
- **Arc: elementwise ops and `sum` are at parity or ahead, matmul is at 0.75-1.1x of PyTorch (depending on
  the iGPU's power state, see pitfall 6), and a training step is at 0.84-0.87x.** The same Arc started the
  first day at 0.19x, 0.23x and 0.26x for add, matmul and the step, then 0.55x for matmul and the step on
  the second. What did it, in the order it paid off:
  - A caching allocator (every `a + b` used to create and commit a fresh 256 MB buffer): `add` 17 -> 80+ GB/s.
  - **A rewritten matmul schedule: 2.3 -> 3.4 TFLOP/s (0.53x -> ~0.8x).** Found by disassembling the
    compiled kernel (`tools/isa.py` in the xpu plugin repo, using Intel's IGA from the driver store) rather
    than by trying variants of the source. The listing showed the B loads issued right before the FMAs that
    use them, so every k step waited on a full cache round trip. Three changes: B loads now run two steps
    ahead through a small register ring; work-groups run in a grouped order so tiles in flight together
    share a compact block of A and B (plain raster order streams all of B for every wave: this alone is
    worth 1.5-1.7x at 4096); and offsets are 32-bit (64-bit index math was ~7 scalar instructions per A
    load and ~30 per B load; on its own this changed nothing, which is why it had been ruled out before).
  - `MATMUL_T`: the backward pass no longer materializes a transposed copy of every weight. For `A B^T`
    with a small batch the plugin computes `C^T = B A^T` and transposes the small activation and the small
    result instead. Training step +12% (88 -> 101 steps/s, interleaved A/B runs).
  - float4 elementwise kernels: add 0.85x -> 1.08x, exp 0.77 -> 0.96, relu 0.74 -> 0.98, tanh 0.69 -> 0.96.
  - The fast matmul now covers any shape: a K tail runs in the kernel and leftover rows and columns go to
    the general kernel (1000³: 1.7 -> 1.1 ms; the rest is unaligned 4000-byte row strides).
  - A fused `AXPY` optimizer update (SGD step 4.2 -> 1.8 ms), a tiled local-memory transpose (42 -> 61-64
    GB/s) and 4-wide loads in the row `sum` (68 -> 82-87 GB/s).
- **Where the Arc matmul stands, and why it stops there.** Removing every load from the new kernel leaves
  4.7-4.9 TFLOP/s of FMA throughput (a ceiling for this instruction stream), so the remaining gap is load
  traffic: a 16 x 32 tile moves 192 bytes per 1024 FLOP, about 75% of an Xe core's L1 bandwidth at full
  FMA rate. A 32-row tile halves that, but needs the 256-register mode (acc 128 + A 64 registers), and the
  compiler spills there (61-226 scratch operations in the loop). A SIMD8 variant (half the A registers) was
  no better (3.1 TFLOP/s). What PyTorch's oneMKL kernel has that this one cannot: 2-D block loads, which
  the driver does not expose to OpenCL. Ruled out by measurement along the way: the scalar broadcast of A
  (same speed without it), A loads (7% of the time), compiler flags (`-cl-fast-relaxed-math`, large-GRF
  modes: no change), and L1 traffic alone (a taller tile that did not spill would be the test).
  `tools/matmul_bench.py` and `tools/matmul_variants.cl` reproduce the search.
- **CUDA, after the same kind of per-op comparison** (`ops_compare.py`): the 2-D transpose copy went from
  0.56x to 0.93x with a tiled kernel, and a bias-style broadcast add from 0.70x to 1.12x with a kernel that
  has no per-element division. Everything else measured is within 10% of PyTorch (reductions 0.91-0.96x).
  Host overhead is not the gap: a CUDA op costs the host 11-17 us against PyTorch's 13-15, and enqueueing a
  whole training step takes 0.3 ms against PyTorch's 0.95 ms.
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
6. **The iGPU's power state moves PyTorch's numbers as much as ours.** On the same day, PyTorch's own XPU
   matmul measured between 2.9 and 4.4 TFLOP/s depending on how long the machine had been busy (shared
   power and thermal budget with the CPU). The Arc ratios are therefore wide ranges, and a result worth
   keeping needs interleaved runs of both libraries, not one after the other.
7. **Measure the same way on both sides of a comparison.** A direct-OpenCL harness reporting the best of
   30 runs read 3.5 TFLOP/s for a kernel that the benchmark's medians put at 3.4 through the plugin and
   at 2.4 before the change; and a timing loop that enqueues without synchronizing queues thousands of
   kernels and measures nothing (`bench.py` syncs every few calls for that reason).
8. **Look at the machine code before theorizing.** Four plausible explanations for the Arc matmul gap
   (broadcast cost, scalar address math, L1 traffic, register spills) were each tested and each changed
   nothing; the compiled listing showed the real one in a minute.
