"""PyTorches: a PyTorch-interoperable tensor/autograd library with plugin backends.

Hardware support is automatic: at import, the runtime loads every device plugin it finds and
keeps the ones whose own hardware test finds usable devices.
"""

import pathlib as _pathlib

from . import _discover

# Bundled plugins plus any installed plugin packages (entry point group "pytorches.plugins").
_discover.locate_plugins(_pathlib.Path(__file__).resolve().parent)

from . import _native  # noqa: E402  (must come after _locate_plugins)
from ._native import (  # noqa: E402,F401
    Tensor,
    calibrate,
    device_info,
    devices,
    from_dlpack,
    full,
    load_plugins,
    plugin_report,
    randn,
    synchronize,
)
from ._native import GraphInfo, strategies  # noqa: E402,F401
from ._report import ModelPlan, Plan, doctor, plan, place, plan_model  # noqa: E402,F401
from . import nn, optim, safetensors  # noqa: E402,F401
from .torch_io import from_torch, from_torch_state_dict, load_torch, to_torch  # noqa: E402,F401


def zeros(shape, device=None, requires_grad=False):
    return full(shape, 0.0, device, requires_grad)


def ones(shape, device=None, requires_grad=False):
    return full(shape, 1.0, device, requires_grad)


__all__ = [
    "Tensor", "from_dlpack", "randn", "zeros", "ones", "full", "devices", "device_info", "load_plugins",
    "synchronize", "calibrate", "plan", "place", "Plan", "plan_model", "ModelPlan", "GraphInfo",
    "strategies", "doctor", "nn", "optim", "safetensors",
    "from_torch", "to_torch", "from_torch_state_dict", "load_torch",
]
