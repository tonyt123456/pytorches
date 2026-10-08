"""Placement strategies through the Python API: planning a model, forcing a strategy, running the
plan, and the error cases. Needs no particular hardware; split tests skip on one-device machines."""

import os
import pathlib
import unittest

os.environ.setdefault(
    "PYTORCHES_PLUGIN_DIR", str(pathlib.Path(__file__).resolve().parents[2] / "plugins" / "bin")
)
import pytorches as pt
from pytorches import nn

DEVICES = pt.devices()
SIZES, BATCH = [16, 32, 32, 8], 4
N_LAYERS = 2 * (len(SIZES) - 1) - 1


def run_steps(model, steps=3):
    x = pt.randn([BATCH, SIZES[0]], model.input_device or "cpu:0", seed=1)
    y = pt.randn([BATCH, SIZES[-1]], model.output_device or "cpu:0", seed=2)
    opt = pt.optim.SGD(model.parameters(), lr=1e-2)
    out = []
    for _ in range(steps):
        opt.zero_grad()
        loss = nn.mse_loss(model(x), y)
        loss.backward()
        opt.step()
        out.append(loss.item())
    return out


def fresh(device):
    nn.manual_seed(0)
    return nn.mlp(SIZES, device)


class Strategies(unittest.TestCase):
    def test_built_in_strategies_are_listed(self):
        self.assertEqual(pt.strategies(), ["single_device", "layer_split"])

    def test_plan_describes_every_layer(self):
        plan = pt.plan_model(nn.mlp_graph(SIZES, BATCH))
        self.assertEqual(len(plan.devices), N_LAYERS)
        self.assertTrue(set(plan.devices) <= set(DEVICES))
        self.assertGreater(plan.est_step_secs, 0)
        self.assertIn(plan.strategy, pt.strategies())
        self.assertEqual(plan.considered[0]["devices"], plan.devices)

    def test_small_model_goes_on_one_device(self):
        plan = pt.plan_model(nn.mlp_graph(SIZES, BATCH))
        self.assertEqual(plan.strategy, "single_device")
        self.assertEqual(len(set(plan.devices)), 1)

    def test_plan_from_a_model_matches_plan_from_a_graph(self):
        a = pt.plan_model(nn.mlp(SIZES), [BATCH, SIZES[0]])
        b = pt.plan_model(nn.mlp_graph(SIZES, BATCH))
        self.assertEqual(a._graph.layers, b._graph.layers)

    def test_explanation_names_the_choice_and_the_devices(self):
        text = str(pt.plan_model(nn.mlp_graph(SIZES, BATCH)))
        for dev in DEVICES:
            self.assertIn(dev, text)
        self.assertIn("chosen:", text)
        self.assertIn("ms per step", text)

    def test_a_named_strategy_is_forced(self):
        plan = pt.plan_model(nn.mlp_graph(SIZES, BATCH), strategy="single_device")
        self.assertEqual(plan.strategy, "single_device")
        self.assertEqual([p["strategy"] for p in plan.considered], ["single_device"])

    def test_forcing_layer_split_uses_several_devices(self):
        if len(DEVICES) < 2:
            self.skipTest("needs two devices")
        plan = pt.plan_model(nn.mlp_graph(SIZES, BATCH), strategy="layer_split")
        self.assertEqual(plan.strategy, "layer_split")
        self.assertGreaterEqual(len(set(plan.devices)), 2)
        # consecutive pieces: a device never reappears after another took over
        seg = [s[0] for s in plan.segments()]
        self.assertEqual(len(seg), len(set(seg)))

    def test_unknown_strategy_is_a_value_error(self):
        with self.assertRaises(ValueError) as cm:
            pt.plan_model(nn.mlp_graph(SIZES, BATCH), strategy="nope")
        self.assertIn("single_device", str(cm.exception))

    def test_a_model_nothing_can_hold_is_a_memory_error_with_reasons(self):
        huge = nn.mlp_graph([400_000, 400_000, 400_000], 1)  # ~1.3 TB of weights
        with self.assertRaises(MemoryError) as cm:
            pt.plan_model(huge)
        self.assertIn("single_device", str(cm.exception))

    def test_empty_model_is_a_value_error(self):
        with self.assertRaises(ValueError):
            pt.plan_model(nn.Sequential(), [1, 1])


class RunningAPlan(unittest.TestCase):
    def test_a_forced_split_trains_like_the_single_device_model(self):
        if len(DEVICES) < 2:
            self.skipTest("needs two devices")
        ref = run_steps(fresh("cpu:0"))
        plan = pt.plan_model(nn.mlp_graph(SIZES, BATCH), strategy="layer_split")
        nn.manual_seed(0)
        model = nn.mlp(SIZES, plan)  # layers are created directly on their devices
        self.assertEqual(model.layer_devices, plan.devices)
        for layer, dev in zip(model.layers, plan.devices):
            self.assertTrue(all(p.device == dev for p in layer.parameters()))
        got = run_steps(model)
        for a, b in zip(got, ref):
            self.assertAlmostEqual(a, b, delta=1e-4 * (1 + abs(b)))

    def test_place_accepts_a_plan(self):
        if len(DEVICES) < 2:
            self.skipTest("needs two devices")
        plan = pt.plan_model(nn.mlp_graph(SIZES, BATCH), strategy="layer_split")
        model = fresh("cpu:0").place(plan)
        self.assertEqual(model.layer_devices, plan.devices)
        ref = run_steps(fresh("cpu:0"))
        for a, b in zip(run_steps(model), ref):
            self.assertAlmostEqual(a, b, delta=1e-4 * (1 + abs(b)))

    def test_mlp_rejects_a_placement_of_the_wrong_length(self):
        with self.assertRaises(ValueError):
            nn.mlp(SIZES, [DEVICES[0]] * 2)

    def test_mlp_graph_matches_a_real_models_graph(self):
        real = nn.mlp(SIZES).graph_info([BATCH, SIZES[0]])
        self.assertEqual(real.layers, nn.mlp_graph(SIZES, BATCH).layers)
        self.assertEqual(real.input_bytes, nn.mlp_graph(SIZES, BATCH).input_bytes)


if __name__ == "__main__":
    unittest.main()
