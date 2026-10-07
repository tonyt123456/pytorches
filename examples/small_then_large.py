"""PyTorches demo: the library picks the hardware, you never name a device.

    python examples/small_then_large.py
    python examples/small_then_large.py --dry-run      # placement decisions only, allocate nothing

One function, `run()`, is called twice with the same steps (each trains for --seconds, default 20) and no device argument:

  1. a small model (~0.6 GiB to train) fits the fastest GPU, so it runs there at full speed;
  2. a large model (~6.5 GiB to train) does not fit the fast GPU's free memory, so the planner
     moves it to the big-memory GPU, which is slower but runs it.

On the author's laptop that is an NVIDIA RTX (8 GB) for the first and the Intel Arc iGPU (borrows
system RAM) for the second. On other hardware the planner makes the same kind of choice.
"""

import argparse
import statistics
import time

import pytorches as pt

BATCH = 32
LR = 1e-3


def hr(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def run(label, sizes, seconds, pause, dry_run):
    """Train an MLP of the given layer sizes for `seconds` of wall time on whatever device the planner picks."""
    need = pt.nn.estimate_mlp_training_bytes(sizes, BATCH)
    params = sum(a * b + b for a, b in zip(sizes, sizes[1:]))
    print(f"{label}: {len(sizes) - 1} layers, {params / 1e6:,.0f}M parameters")
    print(pt.plan(need))
    if dry_run:
        return None

    def build(device):
        model = pt.nn.mlp(sizes, device)
        x = pt.randn([BATCH, sizes[0]], device, seed=1)
        y = pt.randn([BATCH, sizes[-1]], device, seed=2)
        return model, x, y

    t0 = time.time()
    device, (model, x, y) = pt.place(build, need)  # falls back to the next device on out-of-memory
    opt = pt.optim.SGD(model.parameters(), lr=LR)
    print(f"\n  built on {device} ({pt.device_info(device)['name']}) in {time.time() - t0:.1f}s")

    # plan() and place() each benchmark every device, so all GPUs are busy until here. Stay idle
    # for a moment so a utilization graph shows the setup spike and the training run separately.
    print(f"\n  Setup done (the benchmark above ran on every device). Idle for {pause:.0f}s ...")
    time.sleep(pause)
    print(f"\n  >>> TRAINING on {device} for {seconds:.0f}s: only {device} should be busy <<<")
    times = []
    start = last_print = time.time()
    while time.time() - start < seconds:
        t = time.time()
        opt.zero_grad()
        loss = pt.nn.mse_loss(model(x), y)
        loss.backward()
        opt.step()
        value = loss.item()  # also synchronizes the device
        times.append(time.time() - t)
        if time.time() - last_print >= 1.0 or len(times) == 1:
            last_print = time.time()
            print(f"  t={last_print - start:3.1f}s  step {len(times)}: loss {value:.4f}  "
                  f"({times[-1] * 1000:,.0f} ms)")
    print(f"  {len(times)} steps in {time.time() - start:.1f}s")
    print(f"  >>> TRAINING on {device} finished <<<")

    # Skip the first step (kernel compilation / allocator warm-up) when there are enough steps.
    ms = statistics.median(times[1:] or times) * 1000
    tflops = 6 * params * BATCH / (ms / 1000) / 1e12  # forward + backward ~ 6 FLOPs per weight per sample
    return device, ms, tflops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="only print placement decisions")
    ap.add_argument("--seconds", type=float, default=20.0, help="training time per model")
    ap.add_argument("--pause", type=float, default=5.0, help="idle seconds between setup and training, and between models")
    ap.add_argument("--small-width", type=int, default=4096)
    ap.add_argument("--large-width", type=int, default=8192)
    ap.add_argument("--large-layers", type=int, default=12)
    args = ap.parse_args()

    hr("Hardware found")
    pt.doctor()

    hr("1. Small model")
    small = run("small", [args.small_width] * 5, args.seconds, args.pause, args.dry_run)

    if small:
        print(f"\nIdle for {args.pause:.0f}s before the next model ...")
        time.sleep(args.pause)

    hr("2. Large model (same code, same steps)")
    large = run("large", [args.large_width] * (args.large_layers + 1), args.seconds, args.pause, args.dry_run)

    if small and large:
        hr("Summary")
        print(f"  {'model':<8} {'device':<8} {'ms/step':>10} {'TFLOP/s':>9}")
        for name, (dev, ms, tf) in (("small", small), ("large", large)):
            print(f"  {name:<8} {dev:<8} {ms:>10,.0f} {tf:>9.2f}")
        if small[0] != large[0]:
            print(f"\n  No device was named in this script: the small model ran on {small[0]}, "
                  f"and the large one, which {small[0]} could not hold, ran on {large[0]}.")
        else:
            print(f"\n  Both models fit {small[0]} on this machine, so both ran there.")


if __name__ == "__main__":
    main()
