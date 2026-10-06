"""Run the same workloads in PyTorches and PyTorch on every device both can use, and print a table.

    python benchmarks/compare.py
    python benchmarks/compare.py --torch-python C:\\path\\to\\cuda-venv\\Scripts\\python.exe

PyTorch can only use the devices its build supports (a CUDA wheel for cuda, an XPU wheel for xpu), so
point --torch-python at an interpreter with the build you want to compare against. Pass it more than
once as `--torch-python device=path` to use a different PyTorch per device, e.g.
    --torch-python cuda=C:\\cuda-venv\\Scripts\\python.exe --torch-python xpu=C:\\xpu-venv\\Scripts\\python.exe
"""

import argparse
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent


def run(python, lib, device, env_extra=None):
    import os

    env = dict(os.environ)
    env.setdefault("PYTORCHES_PLUGIN_DIR", str(ROOT / "plugins" / "bin"))
    env["PYTHONIOENCODING"] = "utf8"
    p = subprocess.run(
        [python, str(HERE / "bench.py"), "--lib", lib, "--device", device],
        capture_output=True, text=True, env=env,
    )
    rows = []
    for line in p.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            rows.append(json.loads(line))
    if p.returncode != 0 and not rows:
        msg = (p.stderr.strip().splitlines() or ["failed"])[-1]
        return None, msg
    return rows, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pytorches-python", default=sys.executable, help="interpreter with pytorches installed")
    ap.add_argument("--torch-python", action="append", default=[],
                    help="interpreter with PyTorch, or DEVICE=interpreter (repeatable)")
    ap.add_argument("--devices", default="cpu,cuda,xpu")
    args = ap.parse_args()

    torch_pythons = {}
    default_torch = None
    for item in args.torch_python:
        if "=" in item and not item[1:3] == ":\\":  # DEVICE=path (not a bare drive path like C:\)
            dev, path = item.split("=", 1)
            torch_pythons[dev] = path
        else:
            default_torch = item
    default_torch = default_torch or sys.executable

    results, notes = {}, []
    for device in args.devices.split(","):
        for lib, python in (("pytorches", args.pytorches_python), ("torch", torch_pythons.get(device, default_torch))):
            print(f"running {lib:<9} on {device} ...", flush=True)
            rows, err = run(python, lib, device)
            if rows is None:
                notes.append(f"{lib} on {device}: {err}")
                continue
            for r in rows:
                results.setdefault((device, r["workload"]), {})[lib] = r

    print()
    header = f"{'device':<5} {'workload':<28} {'PyTorches':>22} {'PyTorch':>22} {'PyTorches / PyTorch':>20}"
    print(header)
    print("-" * len(header))
    last = None
    for (device, workload), libs in results.items():
        if last and last != device:
            print()
        last = device

        def cell(r):
            return "n/a" if r is None else f"{r['metric']:>10,.1f} {r['unit']:<8} "

        a, b = libs.get("pytorches"), libs.get("torch")
        ratio = ""
        if a and b:
            ratio = f"{b['ms'] / a['ms']:.2f}x speed" if a["ms"] and b["ms"] else ""
        print(f"{device:<5} {workload:<28} {cell(a):>22} {cell(b):>22} {ratio:>20}")
    print("\n(>1.00x means PyTorches is faster; <1.00x means PyTorch is faster. Median of repeated runs.)")
    for n in notes:
        print(f"note: {n}")


if __name__ == "__main__":
    main()
