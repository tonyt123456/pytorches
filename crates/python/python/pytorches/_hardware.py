"""Hardware detection that does not depend on any plugin, so `doctor()` can say what is present
even when no plugin for it is installed. Reads the OS's device list only; loads no vendor driver."""

import json
import pathlib
import platform
import re
import subprocess

# PCI vendor id -> (vendor, name of the plugin that drives it).
VENDORS = {
    0x10DE: ("nvidia", "cuda"),
    0x1002: ("amd", "rocm"),
    0x8086: ("intel", "xpu"),
    0x106B: ("apple", "metal"),
}

# PCI base class + subclass for display controllers (VGA, 3D, other display).
_DISPLAY_CLASSES = {0x0300, 0x0302, 0x0380}


def _gpu(vendor_id, name, source):
    vendor, plugin = VENDORS.get(vendor_id, (f"pci:{vendor_id:04x}", None))
    return {"vendor": vendor, "plugin": plugin, "name": name, "source": source}


def _linux_pci(root=pathlib.Path("/sys/bus/pci/devices")):
    gpus = []
    if not root.is_dir():
        return gpus
    for dev in sorted(root.iterdir()):
        try:
            cls = int((dev / "class").read_text().strip(), 16) >> 8
            if cls not in _DISPLAY_CLASSES:
                continue
            vid = int((dev / "vendor").read_text().strip(), 16)
            did = (dev / "device").read_text().strip()
        except (OSError, ValueError):
            continue
        gpus.append(_gpu(vid, f"PCI device {did} at {dev.name}", "pci"))
    return gpus


def _windows_video(run=subprocess.run):
    cmd = [
        "powershell", "-NoProfile", "-NonInteractive", "-Command",
        "Get-CimInstance Win32_VideoController | Select-Object Name,PNPDeviceID | ConvertTo-Json -Compress",
    ]
    try:
        out = run(cmd, capture_output=True, text=True, timeout=20, check=True).stdout
        data = json.loads(out) if out.strip() else []
    except (OSError, subprocess.SubprocessError, ValueError):
        return []
    return _parse_windows_video(data)


def _parse_windows_video(data):
    if isinstance(data, dict):  # ConvertTo-Json emits a bare object for a single adapter
        data = [data]
    gpus = []
    for entry in data:
        m = re.search(r"VEN_([0-9A-Fa-f]{4})", entry.get("PNPDeviceID") or "")
        if m:  # virtual adapters (remote desktop, basic display) have no PCI vendor id
            gpus.append(_gpu(int(m.group(1), 16), entry.get("Name") or "unknown adapter", "wmi"))
    return gpus


def _macos():
    if platform.machine() == "arm64":
        return [_gpu(0x106B, "Apple silicon GPU", "platform")]
    return []


def detect_hardware():
    """GPUs the OS reports, as dicts with `vendor`, `plugin` (the plugin that drives it, if known),
    `name` and `source`. Empty when nothing can be detected."""
    system = platform.system()
    if system == "Linux":
        return _linux_pci()
    if system == "Windows":
        return _windows_video()
    if system == "Darwin":
        return _macos()
    return []


def diagnose(hardware, report):
    """Match detected hardware against what the loader did.

    `report` is `plugin_report()`: `(file, loaded_plugin_or_None, error_or_None)` per candidate.
    Returns `(hardware_entry, status, message)` with status "ok", "skipped" (a plugin file exists
    but did not load) or "missing" (no plugin file at all).
    """
    loaded = {name for _, name, _ in report if name}
    out = []
    for hw in hardware:
        plugin = hw["plugin"]
        if plugin is None:
            out.append((hw, "missing", "no known plugin for this vendor"))
        elif plugin in loaded:
            out.append((hw, "ok", f"'{plugin}' plugin loaded"))
        else:
            candidates = [(f, e) for f, _, e in report if f"pytorches_plugin_{plugin}" in f]
            if candidates:
                file, err = candidates[0]
                reason = (err or "unknown reason").split(file + ": ", 1)[-1]
                out.append((hw, "skipped", f"'{plugin}' plugin found but not loaded: {reason}"))
            else:
                out.append((hw, "missing", f"no '{plugin}' plugin installed (by convention, package pytorches-{plugin})"))
    return out
