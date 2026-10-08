"""Planner invariants that do not depend on which hardware is present."""

import os
import pathlib
import unittest

os.environ.setdefault(
    "PYTORCHES_PLUGIN_DIR", str(pathlib.Path(__file__).resolve().parents[2] / "plugins" / "bin")
)
import pytorches as pt
from pytorches import _native

GIB = 2**30


def host_free():
    return _native.device_info("cpu:0")["free_memory"]


class SharedMemory(unittest.TestCase):
    def test_device_info_reports_the_flag(self):
        for dev in pt.devices():
            self.assertIn("shared_host_memory", _native.device_info(dev))
        self.assertFalse(_native.device_info("cpu:0")["shared_host_memory"])

    def test_shared_device_is_never_planned_above_host_free(self):
        p = pt.plan(GIB)
        for c in p.candidates:
            if c["shared_host_memory"]:
                # A little slack: host free moves between the two reads.
                self.assertLessEqual(c["free_memory"], host_free() + GIB // 4, c["device"])

    def test_dedicated_device_is_not_capped(self):
        p = pt.plan(GIB)
        for c in p.candidates:
            if c["kind"] == 1:  # cuda
                self.assertEqual(c["free_memory"], _native.device_info(c["device"])["free_memory"])

    def test_paging_warning_when_most_of_host_memory_is_needed(self):
        p = pt.plan(int(host_free() * 0.9))
        # Nothing but the host can hold this, so the plan lands on host memory and says it may page.
        self.assertTrue(p.warnings)

    def test_no_warning_for_a_small_workload(self):
        self.assertEqual(pt.plan(64 * 2**20).warnings, [])


if __name__ == "__main__":
    unittest.main()
