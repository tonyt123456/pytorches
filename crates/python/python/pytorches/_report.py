"""Human-readable reports: hardware detection (`doctor`) and placement decisions (`plan`)."""

from . import _native

_KINDS = {0: "cpu", 1: "cuda", 2: "rocm", 3: "xpu", 4: "npu"}


def _bytes(n):
    if n is None:
        return "unknown"
    return f"{n / 2**30:.1f} GiB" if n >= 2**28 else f"{n / 2**20:.0f} MiB"


def _speed(gflops):
    return f"{gflops / 1000:.2f} TFLOP/s" if gflops >= 1000 else f"{gflops:.1f} GFLOP/s"


class Plan:
    """A placement decision: which device to use for a workload, and why."""

    def __init__(self, raw):
        self._raw = raw
        self.device = raw["chosen"]
        self.reason = raw["reason"]
        self.required_bytes = raw["required_bytes"]
        self.may_oom = raw["may_oom"]
        self.candidates = raw["candidates"]
        self.warnings = raw["warnings"]

    def __str__(self):
        lines = [f"Placement plan: workload needs ~{_bytes(self.required_bytes)}", ""]
        lines.append(f"  {'device':<8} {'name':<34} {'free / total':<20} {'speed':<14} fits")
        for c in self.candidates:
            mem = f"{_bytes(c['free_memory'])} / {_bytes(c['total_memory'])}"
            if c["shared_host_memory"]:
                mem += "*"
            mark = "  <- chosen" if c["device"] == self.device else ""
            lines.append(
                f"  {c['device']:<8} {c['name'][:34]:<34} {mem:<20} {_speed(c['gflops']):<14} "
                f"{'yes' if c['fits'] else 'no'}{mark}"
            )
        if any(c["shared_host_memory"] for c in self.candidates):
            lines.append("  * shares system RAM; free is capped at what the host has free")
        lines += ["", f"  -> {self.device}: {self.reason}"]
        lines += [f"  warning: {w}" for w in self.warnings]
        return "\n".join(lines)

    __repr__ = __str__


def plan(required_bytes):
    """Choose the best device for a workload needing `required_bytes` of device memory."""
    return Plan(_native.plan_placement(int(required_bytes)))


def _secs(s):
    return f"{s * 1000:.1f} ms" if s < 1 else f"{s:.2f} s"


class ModelPlan:
    """Where each layer of a model goes, why, and what it is expected to cost.

    `devices` has one entry per layer; pass the plan to `Sequential.place` or `nn.mlp`.
    `considered` lists every strategy's proposal (fastest first, the first is the chosen one) and
    `declined` the strategies that could not propose and why. Printing it explains the decision.
    """

    def __init__(self, raw, graph):
        self._raw, self._graph = raw, graph
        chosen = raw["chosen"]
        self.devices = chosen["devices"]
        self.strategy = chosen["strategy"]
        self.est_step_secs = chosen["est_step_secs"]
        self.reason = chosen["reason"]
        self.considered = raw["considered"]
        self.declined = raw["declined"]
        self.machine = raw["machine"]

    def segments(self):
        """`[(device, first_layer, last_layer_exclusive)]` for consecutive layers on one device."""
        out = []
        for i, d in enumerate(self.devices):
            if out and out[-1][0] == d:
                out[-1][2] = i + 1
            else:
                out.append([d, i, i + 1])
        return [tuple(s) for s in out]

    def __str__(self):
        g, c = self._graph, self._raw["chosen"]
        free = {m["device"]: m["free_bytes"] for m in self.machine}
        gflop = g.total_flops * 3 / 1e9
        lines = [
            f"Model plan: {len(g.layers)} layers, ~{_bytes(g.total_training_bytes)} to train, "
            f"~{gflop:,.0f} GFLOP per step",
            "",
            f"  {'device':<8} {'name':<34} {'free':<10} {'matmul':<14} memory",
        ]
        for m in self.machine:
            star = "*" if m["shared_host_memory"] else ""
            lines.append(
                f"  {m['device']:<8} {m['name'][:34]:<34} {_bytes(m['free_bytes']) + star:<10} "
                f"{_speed(m['gflops']):<14} {m['gbps']:.0f} GB/s"
            )
        if any(m["shared_host_memory"] for m in self.machine):
            lines.append("  * shares system RAM with the CPU")
        lines += [
            "",
            f"  chosen: {self.strategy}, about {_secs(self.est_step_secs)} per step "
            f"(compute {_secs(c['compute_secs'])}, transfers {_secs(c['transfer_secs'])})",
        ]
        mem = dict(c["per_device"])
        for dev, lo, hi in self.segments():
            names = f"{g.layers[lo][0]} .. {g.layers[hi - 1][0]}" if hi - lo > 1 else g.layers[lo][0]
            lines.append(f"    {dev:<8} layers {lo}..{hi}  ({names})  {_bytes(mem[dev])} of {_bytes(free[dev])} free")
        for after, src, dst, nbytes, secs in c["transfers"]:
            lines.append(
                f"    after layer {after}: {src} -> {dst}, {_bytes(nbytes)} each way, ~{_secs(secs)} per step"
            )
        if c["host_pool_bytes"] and len({d for d in self.devices}) > 1:
            lines.append(f"    system RAM load across the CPU and shared-memory devices: {_bytes(c['host_pool_bytes'])}")
        others = [p for p in self.considered if p is not c and p["devices"] != c["devices"]]
        for p in others:
            lines.append(f"  also: {p['strategy']}, about {_secs(p['est_step_secs'])} per step ({p['reason']})")
        for name, why in self.declined:
            lines.append(f"  not possible: {name}: {why}")
        return "\n".join(lines)

    __repr__ = __str__


def plan_model(model, input_shape=None, strategy=None):
    """Plan how to spread a model over this machine's devices.

    `model` is a `nn.Sequential` (give `input_shape`, e.g. `[batch, features]`) or a `GraphInfo`
    such as `nn.mlp_graph(sizes, batch)`, which describes a model without allocating it. Every
    placement strategy proposes a placement with an estimated step time and the fastest wins;
    `strategy="layer_split"` (see `pytorches.strategies()`) forces one. Raises `MemoryError` if
    nothing can hold the model. Returns a `ModelPlan`.
    """
    graph = model if isinstance(model, _native.GraphInfo) else model.graph_info(input_shape)
    return ModelPlan(_native.plan_model(graph, strategy), graph)


def doctor(benchmark=True):
    """Print what PyTorches found on this machine and which backends it loaded."""
    print("PyTorches doctor")
    print("\nplugins:")
    report = _native.plugin_report()
    if not report:
        print("  (none found; set PYTORCHES_PLUGIN_DIR)")
    for file, loaded, err in report:
        if loaded:
            print(f"  loaded   {loaded:<6} ({file})")
        else:
            print(f"  skipped  {file}: {err.split(': ')[-1]}")
    print("\ndevices:")
    for dev in _native.devices():
        info = _native.device_info(dev)
        mem = f"{_bytes(info['free_memory'])} free / {_bytes(info['total_memory'])}"
        if info["shared_host_memory"]:
            mem += " (shared with system RAM)"
        speed = f"  {_speed(_native.calibrate(dev))}" if benchmark else ""
        print(f"  {dev:<8} {info['name'][:40]:<40} {mem}{speed}")


def place(build, required_bytes, verbose=True):
    """Build a workload on the best device, falling back if that device runs out of memory.

    `build(device)` creates whatever the workload needs on `device` and returns it. The planner
    picks the first device; if `build` raises `MemoryError` (free-memory figures are estimates,
    especially for GPUs that share system RAM), the next device that is expected to fit is tried.

    Returns `(device, result)`.
    """
    p = plan(required_bytes)
    order = [p.device] + [c["device"] for c in p.candidates if c["fits"] and c["device"] != p.device]
    last = None
    for dev in order:
        try:
            return dev, build(dev)
        except MemoryError as exc:
            last = exc
            if verbose:
                print(f"  {dev}: out of memory ({exc}); trying the next device")
    raise last if last else MemoryError("no device can hold this workload")
