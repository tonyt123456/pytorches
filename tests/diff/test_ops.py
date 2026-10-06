"""Differential tests: every op's forward value and gradients must match PyTorch (CPU)."""

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


def randn(*shape):
    return torch.randn(*shape)


def pos(*shape):
    return torch.rand(*shape) + 0.5  # strictly positive, away from 0


def to_pt(t, requires_grad=True):
    return pt.Tensor(t.flatten().tolist(), list(t.shape), requires_grad)


# (name, fn, [input generators]). fn uses only syntax shared by both libraries.
CASES = [
    ("add", lambda a, b: a + b, [lambda: randn(3, 4), lambda: randn(3, 4)]),
    ("add_broadcast_row", lambda a, b: a + b, [lambda: randn(3, 4), lambda: randn(4)]),
    ("add_broadcast_col", lambda a, b: a + b, [lambda: randn(3, 1), lambda: randn(1, 4)]),
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
    ("transpose", lambda a: a.t(), [lambda: randn(3, 4)]),
    ("reshape", lambda a: a.reshape([4, 3]), [lambda: randn(2, 6)]),
    ("scalar_rsub", lambda a: 2.0 - a, [lambda: randn(3)]),
    ("scalar_rdiv", lambda a: 2.0 / a, [lambda: pos(3)]),
    ("chain", lambda a, b: ((a @ b).tanh() * 2.0 + 1.0).exp(), [lambda: randn(3, 4), lambda: randn(4, 2)]),
    ("reuse", lambda a: a * a + a, [lambda: randn(6)]),
]


def flatten(x):
    return x.tolist() if isinstance(x, pt.Tensor) else x.detach().flatten().tolist()


class DiffTests(unittest.TestCase):
    def assert_close(self, got, want, what):
        got_t = torch.tensor(flatten(got))
        want_t = torch.tensor(flatten(want))
        self.assertEqual(got_t.shape, want_t.shape, f"{what}: size mismatch")
        self.assertTrue(
            torch.allclose(got_t, want_t, rtol=1e-4, atol=1e-5),
            f"{what}: max abs diff {(got_t - want_t).abs().max().item()}",
        )


def make_test(fn, gens):
    def test(self):
        ref_in = [g().requires_grad_(True) for g in gens]
        ours_in = [to_pt(t.detach()) for t in ref_in]

        ref_out = fn(*ref_in)
        ours_out = fn(*ours_in)

        self.assertEqual(list(ours_out.shape), list(ref_out.shape), "output shape")
        self.assert_close(ours_out, ref_out, "forward")

        (ref_out * ref_out).sum().backward()
        (ours_out * ours_out).sum().backward()
        for i, (r, o) in enumerate(zip(ref_in, ours_in)):
            self.assertIsNotNone(o.grad, f"input {i} has no grad")
            self.assertEqual(list(o.grad.shape), list(r.grad.shape), f"grad {i} shape")
            self.assert_close(o.grad, r.grad, f"grad[{i}]")

    return test


for name, fn, gens in CASES:
    setattr(DiffTests, f"test_{name}", make_test(fn, gens))


class ReductionTests(unittest.TestCase):
    def test_sum_mean_grad(self):
        r = randn(3, 4).requires_grad_(True)
        o = to_pt(r.detach())
        for reduce in ("sum", "mean"):
            r.grad = None
            o.zero_grad()
            ro, oo = getattr(r, reduce)(), getattr(o, reduce)()
            self.assertAlmostEqual(oo.item(), ro.item(), places=4)
            ro.backward()
            oo.backward()
            self.assertTrue(torch.allclose(torch.tensor(o.grad.tolist()), r.grad.flatten(), atol=1e-6))

    def test_grad_accumulates_across_backward_calls(self):
        o = pt.Tensor([1.0, 2.0], [2], True)
        (o * o).sum().backward()
        (o * o).sum().backward()
        self.assertEqual(o.grad.tolist(), [4.0, 8.0])


class PluginTests(unittest.TestCase):
    def test_cpu_plugin_discovered(self):
        self.assertIn("cpu:0", pt.devices())

    def test_device_info(self):
        info = pt.device_info("cpu:0")
        self.assertEqual(info["name"], "cpu")
        self.assertEqual(info["kind"], 0)

    def test_tensor_device_and_to(self):
        t = pt.Tensor([1.0, 2.0], [2])
        self.assertEqual(t.device, "cpu:0")
        self.assertEqual(t.to("cpu:0").tolist(), [1.0, 2.0])

    def test_scalar_follows_tensor_device(self):
        t = pt.Tensor([1.0, 2.0], [2], device="cpu:0")
        self.assertEqual((t * 3.0).device, "cpu:0")

    def test_unknown_device_errors(self):
        with self.assertRaises(ValueError):
            pt.Tensor([1.0], [1], device="nope:0")
        with self.assertRaises(ValueError):
            pt.Tensor([1.0], [1]).to("cpu:9")

    def test_load_plugins_from_empty_dir(self):
        self.assertEqual(pt.load_plugins(str(pathlib.Path(__file__).parent)), [])


if __name__ == "__main__":
    unittest.main()
