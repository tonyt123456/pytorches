"""One benchmark process: run the same workloads with either PyTorches or PyTorch on one device.

Prints one JSON object per workload, e.g.
  {"lib": "pytorches", "device": "cuda", "workload": "matmul_4096", "ms": 28.1, "metric": 4890.0, "unit": "GFLOP/s"}

Usage:  python bench.py --lib pytorches --device cuda
        python bench.py --lib torch --device xpu
Driven by compare.py, which runs each (lib, device) in its own process so GPU memory, kernel JIT
and caches don't leak between measurements.
"""

import argparse
import json
import statistics
import time

# Problem sizes per device class. Identical for both libraries, so results are comparable.
SIZES = {
    "gpu": dict(matmul=[2048, 4096], vec=64 * 2**20, mlp=([2048] * 4, 256)),
    "cpu": dict(matmul=[512, 1024], vec=8 * 2**20, mlp=([512] * 4, 64)),
}


def median_ms(fn, sync, warmup, iters, warm_seconds):
    """Median wall time of `fn` (synchronized) after warming up for at least `warm_seconds`.

    The time-based warm-up matters on laptop GPUs: they idle at a low clock and take around a second
    of sustained load to reach full speed, so a fixed number of warm-up calls can leave one library
    measured while the GPU is still ramping up and the other at full clocks.
    """
    start = time.perf_counter()
    n = 0
    while n < warmup or time.perf_counter() - start < warm_seconds:
        fn()
        n += 1
        if n % 4 == 0:
            sync()  # don't let an unbounded queue build up while warming
    sync()
    times = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times)


class PyTorchesBackend:
    name = "pytorches"

    def __init__(self, device):
        import pytorches as pt

        self.pt = pt
        self.dev = f"{device}:0"
        if self.dev not in pt.devices():
            raise SystemExit(f"device {self.dev} not available in pytorches ({pt.devices()})")

    def sync(self):
        self.pt.synchronize(self.dev)

    def randn(self, *shape, seed=1):
        return self.pt.randn(list(shape), self.dev, seed=seed)

    def matmul(self, n):
        a, b = self.randn(n, n, seed=1), self.randn(n, n, seed=2)
        return lambda: a @ b

    def add(self, n):
        a, b = self.randn(n, seed=1), self.randn(n, seed=2)
        return lambda: a + b

    def sum(self, n):
        a = self.randn(n, seed=1)
        return lambda: a.sum()

    def mlp_step(self, sizes, batch):
        pt = self.pt
        model = pt.nn.mlp(sizes, self.dev)
        x, y = self.randn(batch, sizes[0], seed=1), self.randn(batch, sizes[-1], seed=2)
        opt = pt.optim.SGD(model.parameters(), lr=1e-3)

        def step():
            opt.zero_grad()
            loss = pt.nn.mse_loss(model(x), y)
            loss.backward()
            opt.step()

        return step


class TorchBackend:
    name = "torch"

    def __init__(self, device):
        import torch

        self.torch = torch
        self.device = device
        self.dev = torch.device(device)
        if device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("this PyTorch build has no CUDA device")
        if device == "xpu" and not torch.xpu.is_available():
            raise SystemExit("this PyTorch build has no XPU device")

    def sync(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        elif self.device == "xpu":
            self.torch.xpu.synchronize()

    def randn(self, *shape, seed=1):
        g = self.torch.Generator(device="cpu").manual_seed(seed)
        return self.torch.randn(*shape, generator=g).to(self.dev)

    def matmul(self, n):
        a, b = self.randn(n, n, seed=1), self.randn(n, n, seed=2)
        return lambda: a @ b

    def add(self, n):
        a, b = self.randn(n, seed=1), self.randn(n, seed=2)
        return lambda: a + b

    def sum(self, n):
        a = self.randn(n, seed=1)
        return lambda: a.sum()

    def mlp_step(self, sizes, batch):
        torch = self.torch
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(torch.nn.ReLU())
        model = torch.nn.Sequential(*layers).to(self.dev)
        x, y = self.randn(batch, sizes[0], seed=1), self.randn(batch, sizes[-1], seed=2)
        opt = torch.optim.SGD(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()

        def step():
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()

        return step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", choices=["pytorches", "torch"], required=True)
    ap.add_argument("--device", choices=["cpu", "cuda", "xpu"], required=True)
    args = ap.parse_args()

    be = PyTorchesBackend(args.device) if args.lib == "pytorches" else TorchBackend(args.device)
    cfg = SIZES["cpu" if args.device == "cpu" else "gpu"]
    warmup, iters, warm_s = (2, 5, 0.3) if args.device == "cpu" else (5, 30, 1.5)

    def emit(workload, ms, metric, unit):
        print(json.dumps({"lib": args.lib, "device": args.device, "workload": workload,
                          "ms": round(ms, 3), "metric": round(metric, 1), "unit": unit}), flush=True)

    for n in cfg["matmul"]:
        ms = median_ms(be.matmul(n), be.sync, warmup, iters, warm_s)
        emit(f"matmul_{n}", ms, 2 * n**3 / (ms / 1000) / 1e9, "GFLOP/s")

    n = cfg["vec"]
    ms = median_ms(be.add(n), be.sync, warmup, iters, warm_s)
    emit(f"add_{n // 2**20}M", ms, 3 * 4 * n / (ms / 1000) / 1e9, "GB/s")
    ms = median_ms(be.sum(n), be.sync, warmup, iters, warm_s)
    emit(f"sum_{n // 2**20}M", ms, 4 * n / (ms / 1000) / 1e9, "GB/s")

    sizes, batch = cfg["mlp"]
    ms = median_ms(be.mlp_step(sizes, batch), be.sync, warmup, iters, warm_s)
    emit(f"mlp_train_step_{sizes[0]}x{len(sizes) - 1}_b{batch}", ms, 1000 / ms, "steps/s")


if __name__ == "__main__":
    main()
