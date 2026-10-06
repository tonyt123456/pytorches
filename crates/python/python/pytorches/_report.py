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

    def __str__(self):
        lines = [f"Placement plan: workload needs ~{_bytes(self.required_bytes)}", ""]
        lines.append(f"  {'device':<8} {'name':<34} {'free / total':<20} {'speed':<14} fits")
        for c in self.candidates:
            mem = f"{_bytes(c['free_memory'])} / {_bytes(c['total_memory'])}"
            mark = "  <- chosen" if c["device"] == self.device else ""
            lines.append(
                f"  {c['device']:<8} {c['name'][:34]:<34} {mem:<20} {_speed(c['gflops']):<14} "
                f"{'yes' if c['fits'] else 'no'}{mark}"
            )
        lines += ["", f"  -> {self.device}: {self.reason}"]
        return "\n".join(lines)

    __repr__ = __str__


def plan(required_bytes):
    """Choose the best device for a workload needing `required_bytes` of device memory."""
    return Plan(_native.plan_placement(int(required_bytes)))


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
