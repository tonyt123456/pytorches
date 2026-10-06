"""PyTorches demo: hardware optimization is automatic.

Run it with no configuration:

    python examples/demo.py              # full demo
    python examples/demo.py --dry-run    # show placement decisions only, allocate nothing
    python examples/demo.py --layers 12 --width 16384 # ~26 GiB: only for machines with lots of free RAM

What it shows
  1. doctor():      which hardware was found and which backends were loaded
  2. speed:         the same small model timed on every device
  3. fits fast:     a model that fits the fast GPU is placed there automatically
  4. too big:       a model that does NOT fit the fast GPU is placed on the big-memory device
"""

import argparse
import time

import pytorches as pt


def hr(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def train(sizes, batch, device, steps, label):
    """Build `sizes` as an MLP on `device`, run SGD steps, print per-step timing."""
    t0 = time.time()
    model = pt.nn.mlp(sizes, device)
    x = pt.randn([batch, sizes[0]], device, seed=1)
    y = pt.randn([batch, sizes[-1]], device, seed=2)
    opt = pt.optim.SGD(model.parameters(), lr=1e-3)
    params = model.num_parameters()
    print(f"  built {label}: {params / 1e6:,.0f}M parameters "
          f"({params * 4 / 2**30:.2f} GiB) on {device} in {time.time() - t0:.1f}s")

    losses, times = [], []
    for step in range(steps):
        t = time.time()
        opt.zero_grad()
        loss = pt.nn.mse_loss(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())  # .item() also synchronizes the device
        times.append(time.time() - t)
        print(f"  step {step + 1}/{steps}: loss {losses[-1]:.4f}  ({times[-1] * 1000:,.0f} ms)")
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="only print placement decisions")
    ap.add_argument("--layers", type=int, default=12, help="layers in the big model")
    ap.add_argument("--width", type=int, default=8192, help="width of the big model")
    ap.add_argument("--steps", type=int, default=3)
    args = ap.parse_args()

    hr("1. Hardware detection")
    pt.doctor()

    if not args.dry_run:
        hr("2. Same model, every device")
        sizes, batch = [2048, 2048, 2048, 2048], 256
        print(f"  MLP {sizes}, batch {batch}: median step time per device")
        for dev in pt.devices():
            ms = sorted(train_quiet(sizes, batch, dev, 5))[2] * 1000
            print(f"    {dev:<8} {pt.device_info(dev)['name'][:38]:<38} {ms:8.1f} ms/step")

    hr("3. A model that fits the fast GPU")
    small = [4096] * 5
    need = pt.nn.estimate_mlp_training_bytes(small, 256)
    plan = pt.plan(need)
    print(plan)
    if not args.dry_run:
        print()
        train(small, 256, plan.device, args.steps, "small model")

    hr("4. A model that does NOT fit the fast GPU")
    big = [args.width] * (args.layers + 1)
    need = pt.nn.estimate_mlp_training_bytes(big, 32)
    plan = pt.plan(need)
    print(plan)
    if not args.dry_run:
        if plan.may_oom:
            print("\n  (no device reports enough free memory; skipping the run)")
        else:
            print()
            train(big, 32, plan.device, args.steps, "big model")
            print(f"\n  Same script, no flags: placement chose {plan.device} on its own.")


def train_quiet(sizes, batch, device, steps):
    model = pt.nn.mlp(sizes, device)
    x = pt.randn([batch, sizes[0]], device, seed=1)
    y = pt.randn([batch, sizes[-1]], device, seed=2)
    opt = pt.optim.SGD(model.parameters(), lr=1e-3)
    times = []
    for _ in range(steps):
        t = time.time()
        opt.zero_grad()
        loss = pt.nn.mse_loss(model(x), y)
        loss.backward()
        opt.step()
        loss.item()
        times.append(time.time() - t)
    return times


if __name__ == "__main__":
    main()
