"""A model that does not fit the fast GPU, run across both GPUs.

    python examples/split_model.py                    # ~9 GiB to train: too big for an 8 GB card
    python examples/split_model.py --width 4096       # smaller, fits one device
    python examples/split_model.py --strategy layer_split

What it shows
  1. the planner describes the model without allocating it, asks every placement strategy, and
     explains its choice (which device runs which layers, what moves between devices, what it costs)
  2. the model is created directly on the planned devices
  3. the same model run entirely on the biggest-memory device, for comparison
  4. both runs give the same losses, and the planned one is faster
"""

import argparse
import statistics
import time

import pytorches as pt


def train(model, x, y, steps):
    opt = pt.optim.SGD(model.parameters(), lr=1e-4)
    losses, times = [], []
    for _ in range(steps):
        t = time.time()
        opt.zero_grad()
        loss = pt.nn.mse_loss(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        for d in set(model.layer_devices):
            pt.synchronize(d)
        times.append(time.time() - t)
    return losses, statistics.median(times[1:])


def build(sizes, batch, placement):
    pt.nn.manual_seed(7)  # same weights whatever the placement
    model = pt.nn.mlp(sizes, placement)
    x = pt.randn([batch, sizes[0]], model.input_device, seed=1)
    y = pt.randn([batch, sizes[-1]], model.output_device, seed=2)
    return model, x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=8192)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--strategy", default=None, help=f"force one of {pt.strategies()}")
    args = ap.parse_args()

    sizes = [args.width] * (args.layers + 1)
    graph = pt.nn.mlp_graph(sizes, args.batch)

    print("1. Plan (nothing allocated yet)\n")
    plan = pt.plan_model(graph, strategy=args.strategy)
    print(plan)

    print("\n2. Run the plan")
    model, x, y = build(sizes, args.batch, plan)
    planned_losses, planned = train(model, x, y, args.steps)
    print(f"   {sorted(set(plan.devices))}: {planned * 1000:.0f} ms/step (estimated {plan.est_step_secs * 1000:.0f})")
    del model, x, y

    print("\n3. The same model on one device")
    single = pt.plan_model(graph, strategy="single_device")
    model, x, y = build(sizes, args.batch, single)
    single_losses, alone = train(model, x, y, args.steps)
    print(f"   {single.devices[0]}: {alone * 1000:.0f} ms/step (estimated {single.est_step_secs * 1000:.0f})")

    print("\n4. Compare")
    worst = max(abs(a - b) / (1 + abs(b)) for a, b in zip(planned_losses, single_losses))
    print(f"   losses, planned : {[round(v, 4) for v in planned_losses[:3]]} ...")
    print(f"   losses, one dev : {[round(v, 4) for v in single_losses[:3]]} ...")
    print(f"   largest relative difference {worst:.1e}")
    print(f"   speed: planned is {alone / planned:.2f}x the speed of the single-device run")


if __name__ == "__main__":
    main()
