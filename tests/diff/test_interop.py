"""Interoperability with PyTorch: DLPack, safetensors, state_dict, and torch.save checkpoints.

Everything is checked against the real PyTorch (and the reference `safetensors` package when
installed), not against PyTorches' own readers.
"""

import gc
import io
import json
import os
import pathlib
import pickle
import struct
import tempfile
import unittest
import zipfile

import torch

os.environ.setdefault(
    "PYTORCHES_PLUGIN_DIR", str(pathlib.Path(__file__).resolve().parents[2] / "plugins" / "bin")
)
import pytorches as pt

try:
    import safetensors.torch as st_ref
except ImportError:  # the reference implementation is optional
    st_ref = None

DEVICES = pt.devices()
torch.manual_seed(0)


def vals(t):
    return t.tolist() if isinstance(t, pt.Tensor) else t.detach().flatten().tolist()


class DLPackTests(unittest.TestCase):
    def test_import_is_zero_copy(self):
        a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        t = pt.from_dlpack(a)
        self.assertEqual(t.shape, [2, 3])
        self.assertEqual(vals(t), vals(a))
        a[0, 0] = 100.0  # visible through the imported tensor: memory is shared
        self.assertEqual(vals(t)[0], 100.0)

    def test_export_is_zero_copy(self):
        p = pt.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
        b = torch.from_dlpack(p)
        self.assertEqual(b.tolist(), [[1.0, 2.0], [3.0, 4.0]])
        b[0, 0] = -1.0
        self.assertEqual(p.tolist()[0], -1.0)

    def test_exported_memory_outlives_source(self):
        p = pt.Tensor([1.0, 2.0, 3.0], [3])
        b = torch.from_dlpack(p)
        del p
        gc.collect()
        self.assertEqual(b.tolist(), [1.0, 2.0, 3.0])

    def test_imported_memory_outlives_source(self):
        t = pt.from_dlpack(torch.full((1000,), 7.0))
        gc.collect()
        self.assertEqual(vals(t)[:3], [7.0, 7.0, 7.0])
        self.assertEqual(sum(vals(t)), 7000.0)

    def test_unconsumed_capsule_is_released(self):
        # Creating and dropping capsules without consuming them must not leak or crash.
        p = pt.Tensor([1.0, 2.0], [2])
        for _ in range(100):
            p.__dlpack__()
        gc.collect()
        self.assertEqual(p.tolist(), [1.0, 2.0])

    def test_views_with_strides_and_offsets(self):
        base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        for view in (base.t(), base[:, 1:], base[1:3, ::2], base[::2].t()):
            self.assertEqual(vals(pt.from_dlpack(view)), vals(view.contiguous()), f"shape {tuple(view.shape)}")

    def test_dtypes_widen_to_float32(self):
        cases = [
            torch.tensor([1.5, -2.25], dtype=torch.float64),
            torch.tensor([1.5, -2.25], dtype=torch.float16),
            torch.tensor([1.5, -2.25], dtype=torch.bfloat16),
            torch.tensor([3, -4], dtype=torch.int64),
            torch.tensor([3, -4], dtype=torch.int32),
            torch.tensor([3, 250], dtype=torch.uint8),
            torch.tensor([True, False]),
        ]
        for c in cases:
            self.assertEqual(vals(pt.from_dlpack(c)), c.float().tolist(), str(c.dtype))

    def test_scalar_and_empty(self):
        s = pt.from_dlpack(torch.tensor(3.5))
        self.assertEqual(s.shape, [])
        self.assertEqual(s.item(), 3.5)
        e = pt.from_dlpack(torch.zeros(0, 4))
        self.assertEqual(e.shape, [0, 4])
        self.assertEqual(vals(e), [])

    def test_requires_grad_source_via_from_torch(self):
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        self.assertEqual(vals(pt.from_torch(x)), [1.0, 2.0])

    def test_non_cpu_export_gives_a_clear_error(self):
        for dev in DEVICES:
            if dev.startswith("cpu"):
                continue
            t = pt.Tensor([1.0, 2.0], [2], device=dev)
            with self.assertRaises(BufferError) as ctx:
                t.__dlpack_device__()
            self.assertIn("to('cpu:0')", str(ctx.exception))
            self.assertEqual(torch.from_dlpack(t.to("cpu:0")).tolist(), [1.0, 2.0])

    def test_roundtrip_helpers_on_every_device(self):
        x = torch.randn(3, 4)
        for dev in DEVICES:
            t = pt.from_torch(x, device=dev)
            self.assertEqual(t.device, dev)
            self.assertEqual(vals(t), vals(x))
            self.assertEqual(vals(pt.to_torch(t)), vals(x))

    @unittest.skipUnless(hasattr(torch, "xpu") and torch.xpu.is_available(), "no torch XPU device")
    def test_producer_on_another_device_is_copied(self):
        x = torch.tensor([1.0, 2.0, 3.0], device="xpu")
        t = pt.from_dlpack(x)
        self.assertEqual(t.device, "cpu:0")
        self.assertEqual(vals(t), [1.0, 2.0, 3.0])

    def test_rejects_non_dlpack_objects(self):
        with self.assertRaises(TypeError):
            pt.from_dlpack([1.0, 2.0])


def _write_safetensors(path, entries, metadata=None):
    """Hand-rolled writer from the format spec, independent of PyTorches' own."""
    header, blob = {}, b""
    for name, (dtype, shape, raw) in entries.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(blob), len(blob) + len(raw)]}
        blob += raw
    if metadata:
        header["__metadata__"] = metadata
    body = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(body)) + body + blob)


class SafetensorsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_roundtrip(self):
        original = {"a": pt.randn([3, 4], seed=1), "b": pt.randn([5], seed=2), "s": pt.Tensor([2.5], [])}
        path = self.dir / "m.safetensors"
        pt.safetensors.save(original, path, metadata={"note": "hi"})
        loaded = pt.safetensors.load(path)
        self.assertEqual(set(loaded), set(original))
        for k in original:
            self.assertEqual(loaded[k].shape, original[k].shape)
            self.assertEqual(vals(loaded[k]), vals(original[k]))
        self.assertEqual(pt.safetensors.metadata(path), {"note": "hi"})

    @unittest.skipIf(st_ref is None, "reference safetensors package not installed")
    def test_reference_implementation_reads_our_files(self):
        original = {"w": pt.randn([4, 6], seed=3), "b": pt.randn([6], seed=4)}
        path = self.dir / "ours.safetensors"
        pt.safetensors.save(original, path, metadata={"k": "v"})
        theirs = st_ref.load_file(str(path))
        for k, t in original.items():
            self.assertEqual(theirs[k].dtype, torch.float32)
            self.assertEqual(theirs[k].flatten().tolist(), vals(t))

    @unittest.skipIf(st_ref is None, "reference safetensors package not installed")
    def test_we_read_reference_files_in_every_dtype(self):
        tensors = {
            "f32": torch.randn(3, 5),
            "f16": torch.randn(7).half(),
            "bf16": torch.randn(2, 3).bfloat16(),
            "f64": torch.randn(4).double(),
            "i64": torch.tensor([[1, -2], [3, 4]]),
            "u8": torch.tensor([0, 255, 7], dtype=torch.uint8),
            "bool": torch.tensor([True, False, True]),
            "scalar": torch.tensor(1.25),
        }
        path = self.dir / "ref.safetensors"
        st_ref.save_file(tensors, str(path))
        loaded = pt.safetensors.load(path)
        for name, want in tensors.items():
            self.assertEqual(loaded[name].shape, list(want.shape), name)
            self.assertEqual(vals(loaded[name]), want.float().flatten().tolist(), name)

    def test_loads_onto_each_device(self):
        path = self.dir / "d.safetensors"
        pt.safetensors.save({"x": pt.Tensor([1.0, 2.0], [2])}, path)
        for dev in DEVICES:
            t = pt.safetensors.load(path, device=dev)["x"]
            self.assertEqual(t.device, dev)
            self.assertEqual(vals(t), [1.0, 2.0])

    def test_hand_written_f16_file(self):
        raw = struct.pack("<2H", 0x3C00, 0xC000)  # 1.0, -2.0
        path = self.dir / "h.safetensors"
        _write_safetensors(path, {"h": ("F16", [2], raw)})
        self.assertEqual(vals(pt.safetensors.load(path)["h"]), [1.0, -2.0])

    def test_malformed_files_are_rejected(self):
        bad = {
            "too_short": b"abc",
            "huge_header": struct.pack("<Q", 2**40) + b"{}",
            "truncated_header": struct.pack("<Q", 100) + b"{}",
            "not_json": struct.pack("<Q", 4) + b"nope",
        }
        for name, data in bad.items():
            path = self.dir / f"{name}.safetensors"
            path.write_bytes(data)
            with self.assertRaises(ValueError, msg=name):
                pt.safetensors.load(path)

    def test_out_of_range_offsets_and_size_mismatch(self):
        path = self.dir / "o.safetensors"
        body = json.dumps({"x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 99]}}).encode()
        path.write_bytes(struct.pack("<Q", len(body)) + body + b"\0" * 8)
        with self.assertRaises(ValueError):
            pt.safetensors.load(path)
        # shape says 3 floats but only 2 are stored
        path2 = self.dir / "o2.safetensors"
        body = json.dumps({"x": {"dtype": "F32", "shape": [3], "data_offsets": [0, 8]}}).encode()
        path2.write_bytes(struct.pack("<Q", len(body)) + body + b"\0" * 8)
        with self.assertRaises(ValueError):
            pt.safetensors.load(path2)

    def test_unknown_dtype(self):
        path = self.dir / "u.safetensors"
        _write_safetensors(path, {"x": ("F8_E4M3", [1], b"\0")})
        with self.assertRaises(ValueError):
            pt.safetensors.load(path)


def make_torch_mlp(sizes):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)


class StateDictTests(unittest.TestCase):
    sizes = [8, 16, 12, 4]

    def test_keys_and_layouts_match_pytorch(self):
        ref = make_torch_mlp(self.sizes)
        ours = pt.nn.mlp(self.sizes)
        sd = ours.state_dict()
        self.assertEqual(list(sd), list(ref.state_dict()))
        for k, v in ref.state_dict().items():
            self.assertEqual(sd[k].shape, list(v.shape), k)

    def test_pytorch_checkpoint_gives_same_outputs(self):
        ref = make_torch_mlp(self.sizes)
        x = torch.randn(5, self.sizes[0])
        want = ref(x).detach().flatten().tolist()
        for dev in DEVICES:
            ours = pt.nn.mlp(self.sizes, dev)
            ours.load_state_dict(pt.from_torch_state_dict(ref.state_dict(), dev))
            got = ours(pt.from_torch(x, dev)).tolist()
            for g, w in zip(got, want):
                self.assertAlmostEqual(g, w, delta=2e-3 + 2e-3 * abs(w), msg=dev)

    def test_our_checkpoint_loads_into_pytorch(self):
        ours = pt.nn.mlp(self.sizes)
        ref = make_torch_mlp(self.sizes)
        ref.load_state_dict({k: pt.to_torch(v) for k, v in ours.state_dict().items()})
        x = torch.randn(5, self.sizes[0])
        got = ref(x).detach().flatten().tolist()
        want = ours(pt.from_torch(x)).tolist()
        for g, w in zip(got, want):
            self.assertAlmostEqual(g, w, delta=1e-4 + 1e-4 * abs(w))

    def test_training_updates_do_not_corrupt_exported_weights(self):
        ours = pt.nn.mlp([4, 8, 2])
        before = {k: vals(v) for k, v in ours.state_dict().items()}
        opt = pt.optim.SGD(ours.parameters(), lr=0.1)
        x, y = pt.randn([6, 4], seed=1), pt.randn([6, 2], seed=2)
        pt.nn.mse_loss(ours(x), y).backward()
        opt.step()
        after = ours.state_dict()
        self.assertTrue(any(vals(after[k]) != before[k] for k in before))
        again = pt.nn.mlp([4, 8, 2])
        again.load_state_dict(after)
        for k, v in again.state_dict().items():
            self.assertEqual(vals(v), vals(after[k]), k)

    def test_safetensors_roundtrip_of_a_model(self):
        ours = pt.nn.mlp(self.sizes)
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "model.safetensors"
            pt.safetensors.save(ours.state_dict(), path)
            fresh = pt.nn.mlp(self.sizes)
            fresh.load_state_dict(pt.safetensors.load(path))
        x = pt.randn([3, self.sizes[0]], seed=5)
        self.assertEqual(ours(x).tolist(), fresh(x).tolist())

    def test_mismatches_are_reported(self):
        ours = pt.nn.mlp([4, 8, 2])
        sd = ours.state_dict()
        missing = dict(sd)
        missing.pop("0.bias")
        with self.assertRaises(KeyError) as ctx:
            ours.load_state_dict(missing)
        self.assertIn("0.bias", str(ctx.exception))
        extra = dict(sd, bogus=pt.zeros([1]))
        with self.assertRaises(KeyError):
            ours.load_state_dict(extra)
        ours.load_state_dict(extra, strict=False)  # tolerated when asked
        wrong = dict(sd, **{"0.weight": pt.zeros([4, 8])})  # [in, out] instead of PyTorch's [out, in]
        with self.assertRaises(ValueError):
            ours.load_state_dict(wrong)


class TorchCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_loads_a_torch_save_state_dict(self):
        ref = make_torch_mlp([6, 10, 3])
        path = self.dir / "m.pt"
        torch.save(ref.state_dict(), path)
        loaded = pt.load_torch(path)
        self.assertEqual(list(loaded), list(ref.state_dict()))
        for k, v in ref.state_dict().items():
            self.assertEqual(loaded[k].shape, list(v.shape))
            self.assertEqual(vals(loaded[k]), vals(v), k)

    def test_checkpoint_to_working_model_without_torch_objects(self):
        ref = make_torch_mlp([6, 10, 3])
        path = self.dir / "m.pt"
        torch.save(ref.state_dict(), path)
        ours = pt.nn.mlp([6, 10, 3])
        ours.load_state_dict(pt.load_torch(path))
        x = torch.randn(4, 6)
        want = ref(x).detach().flatten().tolist()
        for g, w in zip(ours(pt.from_torch(x)).tolist(), want):
            self.assertAlmostEqual(g, w, delta=1e-4 + 1e-4 * abs(w))

    def test_dtypes_offsets_and_shared_storage(self):
        base = torch.arange(10, dtype=torch.float32)
        tensors = {
            "slice": base[3:8],  # storage offset
            "same_storage": base[:5],  # shares storage with "slice"
            "half": torch.randn(4).half(),
            "double": torch.randn(3).double(),
            "int": torch.tensor([[1, 2], [3, 4]]),
            "scalar": torch.tensor(9.0),
        }
        path = self.dir / "d.pt"
        torch.save(tensors, path)
        loaded = pt.load_torch(path)
        for k, want in tensors.items():
            self.assertEqual(vals(loaded[k]), want.float().flatten().tolist(), k)

    def test_loads_onto_a_device(self):
        path = self.dir / "dev.pt"
        torch.save({"w": torch.tensor([1.0, 2.0])}, path)
        for dev in DEVICES:
            self.assertEqual(pt.load_torch(path, device=dev)["w"].device, dev)

    def test_malicious_pickle_is_refused_and_not_executed(self):
        marker = self.dir / "pwned.txt"

        class Evil:
            def __reduce__(self):
                return (open, (str(marker), "w"))

        path = self.dir / "evil.pt"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("archive/data.pkl", pickle.dumps({"w": Evil()}, protocol=2))
        with self.assertRaises(pickle.UnpicklingError):
            pt.load_torch(path)
        self.assertFalse(marker.exists(), "the pickle was executed")

    def test_other_dangerous_globals_are_refused(self):
        for target in ("os.system", "builtins.eval", "subprocess.Popen"):
            module, name = target.rsplit(".", 1)
            payload = b"\x80\x02c" + module.encode() + b"\n" + name.encode() + b"\nq\x00."
            path = self.dir / f"{name}.pt"
            with zipfile.ZipFile(path, "w") as zf:
                zf.writestr("archive/data.pkl", payload)
            with self.assertRaises(pickle.UnpicklingError, msg=target):
                pt.load_torch(path)

    def test_non_contiguous_tensors_are_rejected_clearly(self):
        path = self.dir / "t.pt"
        torch.save({"t": torch.randn(3, 4).t()}, path)
        with self.assertRaises(ValueError) as ctx:
            pt.load_torch(path)
        self.assertIn("contiguous", str(ctx.exception))

    def test_not_a_checkpoint(self):
        path = self.dir / "junk.pt"
        path.write_bytes(b"not a zip file")
        with self.assertRaises(Exception):
            pt.load_torch(path)
        with zipfile.ZipFile(self.dir / "nodata.pt", "w") as zf:
            zf.writestr("hello.txt", "x")
        with self.assertRaises(ValueError):
            pt.load_torch(self.dir / "nodata.pt")


if __name__ == "__main__":
    unittest.main()
