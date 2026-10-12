"""Unit tests for plugin discovery (entry points) and plugin-independent hardware detection."""

import os
import pathlib
import sys
import tempfile
import unittest

from pytorches import _discover, _hardware


def make_plugin_package(root, dist, module):
    """An installed-looking package: `module/` holding a library, plus dist-info declaring the entry
    point. Its `__init__` raises, so any import of it fails the test."""
    pkg = root / module
    pkg.mkdir()
    (pkg / "__init__.py").write_text("raise RuntimeError('must not be imported')\n")
    (pkg / "pytorches_plugin_fake.dll").write_bytes(b"")
    info = root / f"{dist}-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {dist}\nVersion: 1.0\n")
    (info / "entry_points.txt").write_text(f"[{_discover.ENTRY_POINT_GROUP}]\nfake = {module}\n")
    return pkg


class EntryPointDiscovery(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        sys.path.insert(0, self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(sys.path.remove, self.tmp.name)

    def test_finds_plugin_package_without_importing_it(self):
        pkg = make_plugin_package(self.root, "pytorches-fake", "pt_fake_plugin")
        found = dict(_discover.entry_point_plugins())
        self.assertEqual(found["fake"], [pkg])
        self.assertNotIn("pt_fake_plugin", sys.modules)

    def test_entry_point_whose_module_is_missing_is_ignored(self):
        info = self.root / "pytorches-ghost-1.0.dist-info"
        info.mkdir()
        (info / "METADATA").write_text("Metadata-Version: 2.1\nName: pytorches-ghost\nVersion: 1.0\n")
        (info / "entry_points.txt").write_text(f"[{_discover.ENTRY_POINT_GROUP}]\nghost = no_such_module_xyz\n")
        self.assertNotIn("ghost", dict(_discover.entry_point_plugins()))

    def test_bundled_dir_comes_first_and_duplicates_collapse(self):
        pkg = make_plugin_package(self.root, "pytorches-fake", "pt_fake_plugin")
        package_dir = self.root / "pytorches"
        (package_dir / "plugins").mkdir(parents=True)
        dirs = _discover.plugin_dirs(package_dir)
        self.assertEqual(dirs[0], package_dir / "plugins")
        self.assertIn(pkg, dirs)
        keys = [os.path.normcase(os.path.abspath(d)) for d in dirs]
        self.assertEqual(len(keys), len(set(keys)))

    def test_user_set_plugin_dir_is_authoritative(self):
        make_plugin_package(self.root, "pytorches-fake", "pt_fake_plugin")
        old = os.environ.get("PYTORCHES_PLUGIN_DIR")

        def restore():
            if old is None:
                os.environ.pop("PYTORCHES_PLUGIN_DIR", None)
            else:
                os.environ["PYTORCHES_PLUGIN_DIR"] = old

        self.addCleanup(restore)
        os.environ["PYTORCHES_PLUGIN_DIR"] = "user-choice"
        _discover.locate_plugins(self.root / "pytorches")
        self.assertEqual(os.environ["PYTORCHES_PLUGIN_DIR"], "user-choice")
        del os.environ["PYTORCHES_PLUGIN_DIR"]
        _discover.locate_plugins(self.root / "pytorches")
        self.assertIn("pt_fake_plugin", os.environ["PYTORCHES_PLUGIN_DIR"])


class HardwareDetection(unittest.TestCase):
    def test_linux_pci_picks_display_controllers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            for name, cls, vendor, device in [
                ("0000_01_00.0", "0x030000", "0x10de", "0x2bb1"),  # NVIDIA VGA
                ("0000_00_02.0", "0x038000", "0x8086", "0x7d55"),  # Intel display
                ("0000_03_00.0", "0x020000", "0x8086", "0x15f3"),  # network card: ignored
                ("0000_04_00.0", "0x030200", "0x1002", "0x74a1"),  # AMD 3D controller
            ]:
                d = root / name
                d.mkdir()
                (d / "class").write_text(cls + "\n")
                (d / "vendor").write_text(vendor + "\n")
                (d / "device").write_text(device + "\n")
            gpus = _hardware._linux_pci(root)
        self.assertEqual(
            [(g["vendor"], g["plugin"]) for g in gpus],
            [("intel", "xpu"), ("nvidia", "cuda"), ("amd", "rocm")],
        )
        self.assertEqual(_hardware._linux_pci(pathlib.Path("/no/such/dir")), [])

    def test_windows_video_parsing(self):
        data = [
            {"Name": "NVIDIA RTX PRO 1000", "PNPDeviceID": r"PCI\VEN_10DE&DEV_2BB1&SUBSYS_0"},
            {"Name": "Intel(R) Arc(TM) 140T", "PNPDeviceID": r"PCI\VEN_8086&DEV_7D51"},
            {"Name": "Microsoft Remote Display Adapter", "PNPDeviceID": r"SWD\REMOTEDISPLAY\X"},
        ]
        gpus = _hardware._parse_windows_video(data)
        self.assertEqual([g["plugin"] for g in gpus], ["cuda", "xpu"])
        self.assertEqual(len(_hardware._parse_windows_video(data[0])), 1)  # a bare object, not a list

        def no_powershell(*args, **kwargs):
            raise OSError("no powershell")

        self.assertEqual(_hardware._windows_video(run=no_powershell), [])

    def test_unknown_vendor_has_no_plugin(self):
        gpu = _hardware._gpu(0x1234, "mystery", "pci")
        self.assertIsNone(gpu["plugin"])
        [(_, status, _)] = _hardware.diagnose([gpu], [])
        self.assertEqual(status, "missing")

    def test_diagnose_distinguishes_loaded_skipped_and_missing(self):
        hw = [_hardware._gpu(0x10DE, "NVIDIA", "t"), _hardware._gpu(0x1002, "AMD", "t"), _hardware._gpu(0x8086, "Intel", "t")]
        report = [
            ("pytorches_plugin_cuda.dll", "cuda", None),
            ("pytorches_plugin_rocm.dll", None, r"C:\p\pytorches_plugin_rocm.dll: plugin 'rocm' found no devices"),
        ]
        result = {h["plugin"]: (s, m) for h, s, m in _hardware.diagnose(hw, report)}
        self.assertEqual(result["cuda"][0], "ok")
        self.assertEqual(
            result["rocm"], ("skipped", "'rocm' plugin found but not loaded: plugin 'rocm' found no devices")
        )
        self.assertEqual(result["xpu"][0], "missing")
        self.assertIn("pytorches-xpu", result["xpu"][1])

    def test_detect_hardware_never_raises(self):
        self.assertIsInstance(_hardware.detect_hardware(), list)


if __name__ == "__main__":
    unittest.main()
