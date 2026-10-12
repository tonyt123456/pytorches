"""Per-op comparison of PyTorches and PyTorch on one device: finds the individual ops that are slow.

    python benchmarks/ops_compare.py --device xpu  --pytorches-python PY --torch-python PY
    python benchmarks/ops_compare.py --device cuda --pytorches-python PY --torch-python PY_CUDA

Each library runs in its own process (`--worker`); the parent joins the results. Every op is timed with a
synchronize around it after a time-based warm-up, so the numbers include launch overhead, like real use.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

N_VEC = 32 * 2**20  # 32M floats = 128 MiB
SQ = 4096  # square matrix side for transposes / matmul
B, H = 256, 2048  # an MLP layer's activations


def timed(fn, sync, warm_s):
    t0 = time.perf_counter()
    n = 0
    while n < 3 or time.perf_counter() - t0 < warm_s:
        fn()
        n += 1
        if n % 4 == 0:
            sync()
    sync()
    ts = []
    for _ in range(25):
        s = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - s) * 1000)
    return statistics.median(ts)


def workloads_pytorches(device):
    import pytorches as pt

    dev = f"{device}:0"
    r = lambda *shape, seed=1: pt.randn(list(shape), dev, seed=seed)
    sync = lambda: pt.synchronize(dev)
    v, w = r(N_VEC), r(N_VEC, seed=2)
    m, m2 = r(SQ, SQ), r(SQ, SQ, seed=2)
    x, wt, bias = r(B, H), r(H, H, seed=2), r(H, seed=3)
    xg = r(B, H, seed=4).requires_grad_(True)
    big = r(B, H)
    a1000, b1000 = r(1000, 1000), r(1000, 1000, seed=2)
    two = pt.Tensor([2.0], [1], device=dev)
    ops = {
        "add (contiguous)": lambda: v + w,
        "mul (contiguous)": lambda: v * w,
        "exp": lambda: v.exp(),
        "relu": lambda: v.relu(),
        "tanh": lambda: v.tanh(),
        "add bias (broadcast row)": lambda: x + bias,
        "mul scalar-broadcast": lambda: v * two,
        "sum (all, 32M)": lambda: v.sum(),
        "mean (all, 32M)": lambda: v.mean(),
        f"transpose copy {SQ}²": lambda: m.t().contiguous() if hasattr(m, "contiguous") else m.t() + 0,
        f"matmul {SQ}²": lambda: m @ m2,
        f"matmul [{B}x{H}]@[{H}x{H}]": lambda: x @ wt,
        f"matmul A@B^T [{B}x{H}]": lambda: x @ wt.t(),
        f"matmul A^T@B [{B}x{H}]": lambda: x.t() @ big,
        "matmul 1000x1000 (odd shape)": lambda: a1000 @ b1000,
    }
    return ops, sync


def workloads_torch(device):
    import torch

    dev = device
    torch.manual_seed(0)
    r = lambda *shape: torch.randn(*shape, device=dev)
    sync = {"cuda": torch.cuda.synchronize, "xpu": torch.xpu.synchronize if hasattr(torch, "xpu") else None, "cpu": lambda: None}[device]
    v, w = r(N_VEC), r(N_VEC)
    m, m2 = r(SQ, SQ), r(SQ, SQ)
    x, wt, bias = r(B, H), r(H, H), r(H)
    big = r(B, H)
    a1000, b1000 = r(1000, 1000), r(1000, 1000)
    ops = {
        "add (contiguous)": lambda: v + w,
        "mul (contiguous)": lambda: v * w,
        "exp": lambda: v.exp(),
        "relu": lambda: v.relu(),
        "tanh": lambda: v.tanh(),
        "add bias (broadcast row)": lambda: x + bias,
        "mul scalar-broadcast": lambda: v * torch.tensor([2.0], device=dev),
        "sum (all, 32M)": lambda: v.sum(),
        "mean (all, 32M)": lambda: v.mean(),
        f"transpose copy {SQ}²": lambda: m.t().contiguous(),
        f"matmul {SQ}²": lambda: m @ m2,
        f"matmul [{B}x{H}]@[{H}x{H}]": lambda: x @ wt,
        f"matmul A@B^T [{B}x{H}]": lambda: x @ wt.t(),
        f"matmul A^T@B [{B}x{H}]": lambda: x.t() @ big,
        "matmul 1000x1000 (odd shape)": lambda: a1000 @ b1000,
    }
    return ops, sync


def worker(lib, device):
    ops, sync = (workloads_pytorches if lib == "pytorches" else workloads_torch)(device)
    warm = 0.3 if device == "cpu" else 1.0
    for name, fn in ops.items():
        try:
            ms = timed(fn, sync, warm)
            print(json.dumps({"op": name, "ms": ms}), flush=True)
        except Exception as e:  # report and move on: one unsupported op should not hide the rest
            print(json.dumps({"op": name, "error": f"{type(e).__name__}: {e}"[:120]}), flush=True)


def run(python, lib, device):
    env = dict(os.environ, PYTHONIOENCODING="utf8")
    p = subprocess.run([python, __file__, "--worker", lib, "--device", device], capture_output=True, text=True, env=env)
    rows = {}
    for line in p.stdout.splitlines():
        if line.startswith("{"):
            d = json.loads(line)
            rows[d["op"]] = d
    if not rows:
        print(f"{lib}: no results\n{p.stderr[-600:]}", file=sys.stderr)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--pytorches-python", default=sys.executable)
    ap.add_argument("--torch-python", default=sys.executable)
    ap.add_argument("--worker", choices=["pytorches", "torch"])
    args = ap.parse_args()
    if args.worker:
        return worker(args.worker, args.device)
    a = run(args.pytorches_python, "pytorches", args.device)
    b = run(args.torch_python, "torch", args.device)
    print(f"{'op':<34} {'PyTorches ms':>13} {'PyTorch ms':>11} {'speed vs PyTorch':>17}")
    for op in a:
        x, y = a.get(op, {}), b.get(op, {})
        if "ms" in x and "ms" in y:
            print(f"{op:<34} {x['ms']:>13.3f} {y['ms']:>11.3f} {y['ms'] / x['ms']:>16.2f}x")
        else:
            print(f"{op:<34} {x.get('ms', x.get('error', 'n/a'))!s:>13} {y.get('ms', y.get('error', 'n/a'))!s:>11}")


if __name__ == "__main__":
    main()
