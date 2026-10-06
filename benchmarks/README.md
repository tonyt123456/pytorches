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
| **cuda** | matmul 2048² | 3,297 GFLOP/s | 6,132 GFLOP/s | 0.54x |
| | matmul 4096² | 3,694 GFLOP/s | 5,196 GFLOP/s | 0.71x |
| | add, 64M floats | 267 GB/s | 266 GB/s | **1.01x** |
| | sum, 64M floats | 239 GB/s | 271 GB/s | 0.88x |
| | MLP train step (4×2048, batch 256) | 126 steps/s | 232 steps/s | 0.54x |
| **xpu** (Arc) | matmul 2048² | 1,020 GFLOP/s | 4,425 GFLOP/s | 0.23x |
| | matmul 4096² | 1,035 GFLOP/s | 3,926 GFLOP/s | 0.26x |
| | add, 64M floats | 17 GB/s | 91 GB/s | 0.19x |
| | sum, 64M floats | 66 GB/s | 86 GB/s | 0.77x |
| | MLP train step | 36 steps/s | 136 steps/s | 0.26x |
| **cpu** | matmul 1024² | 22 GFLOP/s | 611 GFLOP/s | 0.04x |
| | add, 8M floats | 8.5 GB/s | 64 GB/s | 0.13x |
| | sum, 8M floats | 1.8 GB/s | 88 GB/s | 0.02x |
| | MLP train step (4×512, batch 64) | 49 steps/s | 443 steps/s | 0.11x |

(Workload sizes are smaller on CPU, identical for both libraries on the same device.)

## What this says

- **PyTorch is faster almost everywhere.** It has years of tuning behind it; PyTorches is early alpha.
- **Memory-bound CUDA ops are at parity** (add 1.01x, sum 0.88x): both reach the card's memory bandwidth.
- **CUDA matmul is 1.4-1.9x behind cuBLAS.** Our kernel is correct, and gives bit-identical results to
  cuBLAS on the same inputs, but it is an untuned 128x128 tiled kernel; cuBLAS reaches ~6 TFLOP/s here.
  PyTorch can go further (~8.5 TFLOP/s on this GPU) by enabling TF32 tensor cores, which we don't use.
- **Arc: ~4x behind on matmul** (oneDNN uses the matrix engines; ours does not), and **5x behind on
  elementwise ops**. The latter is largely a missing caching allocator in the xpu plugin: every `a + b`
  creates and commits a fresh 256 MB buffer, which PyTorch avoids by reusing memory.
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
