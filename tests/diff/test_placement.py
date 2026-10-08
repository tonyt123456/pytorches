"""Placement: describing a model to the planner, pricing a placement, and running a model whose
layers live on different devices. A split model must behave exactly like the same model on one
device (forward, gradients, and training steps), for every pair of devices on this machine."""

import os
import pathlib
import unittest

os.environ.setdefault(
    "PYTORCHES_PLUGIN_DIR", str(pathlib.Path(__file__).resolve().parents[2] / "plugins" / "bin")
)
import pytorches as pt
from pytorches import nn

DEVICES = pt.devices()
SIZES, BATCH = [16, 32, 32, 8], 4  # Linear, ReLU, Linear, ReLU, Linear


def build(placement=None):
    nn.manual_seed(0)
    model = nn.mlp(SIZES, "cpu:0")
    if placement is not None:
        model.place(placement)
    x = pt.randn([BATCH, SIZES[0]], "cpu:0", seed=1)
    y = pt.randn([BATCH, SIZES[-1]], "cpu:0", seed=2)
    return model, x, (y if placement is None else y.to(model.output_device))


def step(model, x, y, lr=1e-2):
    model.zero_grad()
    loss = nn.mse_loss(model(x), y)
    loss.backward()
    grads = [p.grad.to("cpu:0").tolist() for p in model.parameters()]
    pt.optim.SGD(model.parameters(), lr).step()
    return loss.item(), grads


def close(a, b, tol=1e-4):
    return len(a) == len(b) and all(abs(u - v) <= tol * (1 + abs(v)) for u, v in zip(a, b))


def placements():
    n = 2 * (len(SIZES) - 1) - 1
    out = []
    for a in DEVICES:
        for b in DEVICES:
            for k in range(1, n):
                out.append([a] * k + [b] * (n - k))
    out.append([DEVICES[i % len(DEVICES)] for i in range(n)])  # bounces between devices
    out.append([DEVICES[-1 - i % len(DEVICES)] for i in range(n)])
    return out


class GraphInfo(unittest.TestCase):
    def test_layer_costs(self):
        g = nn.mlp([8, 16, 4]).graph_info([2, 8])
        self.assertEqual(g.input_bytes, 4 * 2 * 8)
        names = [l[0] for l in g.layers]
        self.assertEqual(names, ["0:Linear", "1:ReLU", "2:Linear"])
        # (name, param_bytes, out_bytes, flops, training_bytes)
        self.assertEqual(g.layers[0][1:4], (4 * (8 * 16 + 16), 4 * 2 * 16, 2 * 2 * 8 * 16))
        self.assertEqual(g.layers[1][1:4], (0, 4 * 2 * 16, 2 * 16))
        self.assertEqual(g.layers[2][1:4], (4 * (16 * 4 + 4), 4 * 2 * 4, 2 * 2 * 16 * 4))
        self.assertEqual(g.total_param_bytes, 4 * (8 * 16 + 16 + 16 * 4 + 4))
        self.assertEqual(g.total_training_bytes, sum(l[4] for l in g.layers))
        self.assertEqual(g.total_flops, sum(l[3] for l in g.layers))

    def test_wrong_input_shape_is_an_error(self):
        with self.assertRaises(ValueError):
            nn.mlp([8, 16, 4]).graph_info([2, 9])

    def test_graph_info_creates_no_tensors_on_devices(self):
        # Describing a huge model must be free: this would not fit anywhere.
        g = nn.Sequential().graph_info([1, 1])
        self.assertEqual(len(g.layers), 0)


class Pricing(unittest.TestCase):
    def setUp(self):
        self.g = nn.mlp(SIZES).graph_info([BATCH, SIZES[0]])
        self.n = len(self.g.layers)

    def test_uniform_placement_has_no_transfers(self):
        r = self.g.price([DEVICES[0]] * self.n)
        self.assertEqual(r["transfers"], [])
        self.assertEqual(r["transfer_bytes"], 0)
        self.assertEqual(r["per_device"], [(DEVICES[0], self.g.total_training_bytes)])

    def test_split_is_priced_per_device_and_per_boundary(self):
        if len(DEVICES) < 2:
            self.skipTest("needs two devices")
        a, b = DEVICES[0], DEVICES[1]
        r = self.g.price([a, a, b, b, b])
        layers = self.g.layers
        self.assertEqual(r["per_device"], [(a, layers[0][4] + layers[1][4]), (b, sum(l[4] for l in layers[2:]))])
        # one boundary, after layer 1; the activation goes forward and its gradient comes back
        self.assertEqual(r["transfers"], [(1, a, b, 2 * layers[1][2])])
        self.assertEqual(r["transfer_bytes"], 2 * layers[1][2])

    def test_every_bounce_is_a_transfer(self):
        if len(DEVICES) < 2:
            self.skipTest("needs two devices")
        a, b = DEVICES[0], DEVICES[1]
        r = self.g.price([a, b, a, b, a])
        self.assertEqual(len(r["transfers"]), 4)

    def test_host_pool_counts_cpu_and_shared_devices_together(self):
        shared = [d for d in DEVICES if pt.device_info(d)["shared_host_memory"]]
        if not shared:
            self.skipTest("no device shares system RAM")
        s = shared[0]
        r = self.g.price(["cpu:0", "cpu:0", s, s, s])
        self.assertEqual(r["host_pool_bytes"], self.g.total_training_bytes)
        # a dedicated device's share does not touch the pool
        dedicated = [d for d in DEVICES if d != "cpu:0" and not pt.device_info(d)["shared_host_memory"]]
        if dedicated:
            r = self.g.price([dedicated[0], dedicated[0], s, s, s])
            self.assertEqual(r["host_pool_bytes"], sum(l[4] for l in self.g.layers[2:]))

    def test_wrong_length_and_bad_device_are_errors(self):
        with self.assertRaises(ValueError):
            self.g.price([DEVICES[0]])
        with self.assertRaises(ValueError):
            self.g.price(["nope:0"] * self.n)


class Executor(unittest.TestCase):
    def test_parameters_move_to_their_layer_devices(self):
        for devs in placements():
            model, _, _ = build(devs)
            for layer, dev in zip(model.layers, devs):
                for p in layer.parameters():
                    self.assertEqual(p.device, dev)
                    self.assertTrue(p.requires_grad)
            self.assertEqual(model.input_device, devs[0])
            self.assertEqual(model.output_device, devs[-1])

    def test_split_model_matches_single_device(self):
        ref_model, x, y = build()
        ref = [step(ref_model, x, y) for _ in range(3)]
        for devs in placements():
            model, x, y = build(devs)
            for i in range(3):
                loss, grads = step(model, x, y)
                self.assertTrue(close([loss], [ref[i][0]]), f"{devs} step {i}: loss {loss} vs {ref[i][0]}")
                for g, r in zip(grads, ref[i][1]):
                    self.assertTrue(close(g, r), f"{devs} step {i}: gradient mismatch")

    def test_single_string_places_every_layer(self):
        model, x, y = build(DEVICES[-1])
        self.assertEqual(model.layer_devices, [DEVICES[-1]] * 5)
        self.assertTrue(close(step(model, x, y)[0:1], step(build()[0], *build()[1:])[0:1]))

    def test_wrong_number_of_devices_is_an_error(self):
        with self.assertRaises(ValueError):
            nn.mlp(SIZES).place([DEVICES[0]] * 2)

    def test_to_moves_everything_and_unsplits(self):
        model = nn.mlp(SIZES).place([DEVICES[0], DEVICES[-1], DEVICES[0], DEVICES[-1], DEVICES[0]])
        model.to(DEVICES[-1])
        self.assertTrue(all(p.device == DEVICES[-1] for p in model.parameters()))
        self.assertEqual(model.layer_devices, [DEVICES[-1]] * 5)


if __name__ == "__main__":
    unittest.main()
