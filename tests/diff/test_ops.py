"""Differential tests: every op's forward value and gradients must match PyTorch (CPU),
on every device PyTorches detects (cpu, cuda, xpu, ...)."""

import os
import pathlib
import unittest

import torch

# Plugins are shared libraries discovered at import time (build with scripts/build-plugins.ps1).
os.environ.setdefault(
    "PYTORCHES_PLUGIN_DIR", str(pathlib.Path(__file__).resolve().parents[2] / "plugins" / "bin")
)
import pytorches as pt

torch.manual_seed(0)

DEVICES = pt.devices()


def randn(*shape):
    return torch.randn(*shape)


def pos(*shape):
    return torch.rand(*shape) + 0.5  # strictly positive, away from 0


def to_pt(t, device, requires_grad=True):
    return pt.Tensor(t.flatten().tolist(), list(t.shape), requires_grad, device)


# (name, fn, [input generators]). fn uses only syntax shared by both libraries.
CASES = [
    ("add", lambda a, b: a + b, [lambda: randn(3, 4), lambda: randn(3, 4)]),
    ("add_broadcast_row", lambda a, b: a + b, [lambda: randn(3, 4), lambda: randn(4)]),
    ("add_broadcast_col", lambda a, b: a + b, [lambda: randn(3, 1), lambda: randn(1, 4)]),
    ("add_3d", lambda a, b: a + b, [lambda: randn(2, 3, 5), lambda: randn(3, 1)]),
    ("sub", lambda a, b: a - b, [lambda: randn(3, 4), lambda: randn(3, 4)]),
    ("sub_broadcast", lambda a, b: a - b, [lambda: randn(2, 3, 4), lambda: randn(3, 1)]),
    ("mul", lambda a, b: a * b, [lambda: randn(3, 4), lambda: randn(3, 4)]),
    ("mul_broadcast", lambda a, b: a * b, [lambda: randn(2, 3), lambda: randn(3)]),
    ("div", lambda a, b: a / b, [lambda: randn(3, 4), lambda: pos(3, 4)]),
    ("div_broadcast", lambda a, b: a / b, [lambda: randn(3, 4), lambda: pos(4)]),
    ("neg", lambda a: -a, [lambda: randn(5)]),
    ("exp", lambda a: a.exp(), [lambda: randn(3, 4)]),
    ("log", lambda a: a.log(), [lambda: pos(3, 4)]),
    ("relu", lambda a: a.relu(), [lambda: randn(4, 5)]),
    ("tanh", lambda a: a.tanh(), [lambda: randn(4, 5)]),
    ("matmul", lambda a, b: a @ b, [lambda: randn(3, 4), lambda: randn(4, 5)]),
    ("matmul_odd", lambda a, b: a @ b, [lambda: randn(33, 65), lambda: randn(65, 17)]),
    ("matmul_big", lambda a, b: a @ b, [lambda: randn(130, 70), lambda: randn(70, 90)]),
    ("transpose", lambda a: a.t(), [lambda: randn(3, 4)]),
    ("reshape", lambda a: a.reshape([4, 3]), [lambda: randn(2, 6)]),
    ("scalar_rsub", lambda a: 2.0 - a, [lambda: randn(3)]),
    ("scalar_rdiv", lambda a: 2.0 / a, [lambda: pos(3)]),
    ("chain", lambda a, b: ((a @ b).tanh() * 2.0 + 1.0).exp(), [lambda: randn(3, 4), lambda: randn(4, 2)]),
    ("reuse", lambda a: a * a + a, [lambda: randn(6)]),
    ("mlp_layer", lambda x, w, b: (x @ w + b).relu(), [lambda: randn(8, 16), lambda: randn(16, 12), lambda: randn(12)]),
]


def flatten(x):
    return x.tolist() if isinstance(x, pt.Tensor) else x.detach().flatten().tolist()


class DeviceTestBase(unittest.TestCase):
    device = "cpu:0"

    @property
    def rtol(self):
        # Tolerance is relative to the largest magnitude in the result, because float32
        # summation order alone moves near-zero entries of large sums by a few ulps of the max.
        # GPUs use different summation orders / fused ops, so they get a little more slack.
        return 1e-5 if self.device.startswith("cpu") else 5e-4

    def assert_close(self, got, want, what):
        got_t = torch.tensor(flatten(got))
        want_t = torch.tensor(flatten(want))
        self.assertEqual(got_t.shape, want_t.shape, f"{what}: size mismatch")
        scale = max(want_t.abs().max().item(), 1.0) if want_t.numel() else 1.0
        self.assertTrue(
            torch.allclose(got_t, want_t, rtol=self.rtol, atol=self.rtol * scale),
            f"[{self.device}] {what}: max abs diff {(got_t - want_t).abs().max().item()} (scale {scale})",
        )


def make_op_test(fn, gens):
    def test(self):
        ref_in = [g().requires_grad_(True) for g in gens]
        ours_in = [to_pt(t.detach(), self.device) for t in ref_in]

        ref_out = fn(*ref_in)
        ours_out = fn(*ours_in)

        self.assertEqual(ours_out.device, self.device)
        self.assertEqual(list(ours_out.shape), list(ref_out.shape), "output shape")
        self.assert_close(ours_out, ref_out, "forward")

        (ref_out * ref_out).sum().backward()
        (ours_out * ours_out).sum().backward()
        for i, (r, o) in enumerate(zip(ref_in, ours_in)):
            self.assertIsNotNone(o.grad, f"input {i} has no grad")
            self.assertEqual(list(o.grad.shape), list(r.grad.shape), f"grad {i} shape")
            self.assert_close(o.grad, r.grad, f"grad[{i}]")

    return test


def reduction_test(self):
    for reduce in ("sum", "mean"):
        r = randn(3, 4).requires_grad_(True)
        o = to_pt(r.detach(), self.device)
        ro, oo = getattr(r, reduce)(), getattr(o, reduce)()
        self.assertAlmostEqual(oo.item(), ro.item(), places=3)
        ro.backward()
        oo.backward()
        self.assert_close(o.grad, r.grad, f"{reduce} grad")


def accumulate_test(self):
    o = pt.Tensor([1.0, 2.0], [2], True, self.device)
    (o * o).sum().backward()
    (o * o).sum().backward()
    self.assertEqual(o.grad.tolist(), [4.0, 8.0])


def creation_test(self):
    self.assertEqual(pt.full([2, 3], 7.0, self.device).tolist(), [7.0] * 6)
    self.assertEqual(pt.zeros([4], self.device).tolist(), [0.0] * 4)
    self.assertEqual(pt.ones([2, 2], self.device).device, self.device)


def randn_test(self):
    # Golden values from the CPU reference recipe (seed 1234).
    got = pt.randn([4], self.device, seed=1234).tolist()
    for g, w in zip(got, [0.6574781, -0.08203645, -2.2065563, 0.70945626]):
        self.assertAlmostEqual(g, w, places=3)
    big = torch.tensor(pt.randn([50_000], self.device, seed=7).tolist())
    self.assertLess(abs(big.mean().item()), 0.03)
    self.assertLess(abs(big.std().item() - 1.0), 0.03)


def in_place_test(self):
    t = pt.zeros([3], self.device)
    alias = t.detach()
    t.copy_(pt.Tensor([1.0, 2.0, 3.0], [3], False, self.device))
    self.assertEqual(alias.tolist(), [1.0, 2.0, 3.0])


def training_test(self):
    """A few SGD steps on a tiny MLP must reduce the loss."""
    model = pt.nn.mlp([8, 16, 1], self.device)
    opt = pt.optim.SGD(model.parameters(), lr=0.05)
    x = pt.randn([32, 8], self.device, seed=1)
    y = pt.randn([32, 1], self.device, seed=2)
    first = last = None
    for _ in range(40):
        opt.zero_grad()
        loss = pt.nn.mse_loss(model(x), y)
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
        last = loss.item()
    self.assertLess(last, first * 0.9, f"loss did not drop: {first} -> {last}")


def cross_device_test(self):
    t = pt.Tensor([1.0, -2.0, 3.0], [3], True, self.device)
    for other in DEVICES:
        moved = t.to(other)
        self.assertEqual(moved.device, other)
        self.assertEqual(moved.tolist(), [1.0, -2.0, 3.0])
    # gradient flows back through a device move
    for other in DEVICES:
        t.zero_grad()
        (t.to(other) * 2.0).sum().backward()
        self.assertEqual(t.grad.tolist(), [2.0, 2.0, 2.0])


# One test class per detected device, so failures name the backend.
for _dev in DEVICES:
    _ns = {"device": _dev}
    for _name, _fn, _gens in CASES:
        _ns[f"test_{_name}"] = make_op_test(_fn, _gens)
    _ns["test_sum_mean_grad"] = reduction_test
    _ns["test_grad_accumulates"] = accumulate_test
    _ns["test_creation"] = creation_test
    _ns["test_randn_matches_reference"] = randn_test
    _ns["test_in_place_copy"] = in_place_test
    _ns["test_training_reduces_loss"] = training_test
    _ns["test_cross_device_moves"] = cross_device_test
    _cls = type(f"Diff_{_dev.replace(':', '_')}", (DeviceTestBase,), _ns)
    globals()[_cls.__name__] = _cls


class PluginTests(unittest.TestCase):
    def test_cpu_plugin_discovered(self):
        self.assertIn("cpu:0", pt.devices())

    def test_device_info(self):
        info = pt.device_info("cpu:0")
        self.assertEqual(info["name"], "cpu")
        self.assertEqual(info["kind"], 0)

    def test_plugin_report_lists_loaded_plugins(self):
        loaded = {name for _, name, _ in pt.plugin_report() if name}
        self.assertIn("cpu", loaded)

    def test_scalar_follows_tensor_device(self):
        for dev in DEVICES:
            t = pt.Tensor([1.0, 2.0], [2], device=dev)
            self.assertEqual((t * 3.0).device, dev)

    def test_unknown_device_errors(self):
        with self.assertRaises(ValueError):
            pt.Tensor([1.0], [1], device="nope:0")
        with self.assertRaises(ValueError):
            pt.Tensor([1.0], [1]).to("cpu:9")

    def test_load_plugins_from_empty_dir(self):
        self.assertEqual(pt.load_plugins(str(pathlib.Path(__file__).parent)), [])

    def test_plan_picks_a_device(self):
        p = pt.plan(1 << 20)
        self.assertIn(p.device, DEVICES)
        self.assertFalse(p.may_oom)
        self.assertIn("needs", p.reason)

    def test_plan_too_big_for_anything(self):
        p = pt.plan(1 << 60)
        self.assertTrue(p.may_oom)


if __name__ == "__main__":
    unittest.main()
