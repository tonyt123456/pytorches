"""Interoperability with PyTorch: tensor exchange and reading `.pt` checkpoints without torch.

`from_torch` / `to_torch` use DLPack (zero-copy for float32 CPU tensors). `load_torch` reads a
file written by `torch.save` using a restricted unpickler, so loading a checkpoint can never
execute arbitrary code: only the handful of classes needed to rebuild tensors are allowed.
"""

import collections
import io
import pickle
import zipfile

from . import _native

# torch storage class name -> safetensors-style dtype name understood by `tensor_from_buffer`
_STORAGE_DTYPES = {
    "FloatStorage": "F32", "DoubleStorage": "F64", "HalfStorage": "F16", "BFloat16Storage": "BF16",
    "LongStorage": "I64", "IntStorage": "I32", "ShortStorage": "I16", "CharStorage": "I8",
    "ByteStorage": "U8", "BoolStorage": "BOOL",
}
_ITEMSIZE = {"F32": 4, "F64": 8, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


def from_torch(t, device=None):
    """Convert a `torch.Tensor` to a `pytorches.Tensor` (zero-copy for float32 CPU tensors).

    The result shares memory with `t` when both are on the CPU. Use `device` to move it.
    """
    out = _native.from_dlpack(t.detach())
    return out if device is None else out.to(device)


def to_torch(t, device=None):
    """Convert a `pytorches.Tensor` to a `torch.Tensor` (zero-copy when `t` is on the CPU)."""
    import torch

    out = torch.from_dlpack(t.to("cpu:0"))
    return out if device is None else out.to(device)


def from_torch_state_dict(state_dict, device=None):
    """`{name: torch.Tensor}` -> `{name: pytorches.Tensor}`."""
    return {k: from_torch(v, device) for k, v in state_dict.items()}


# ---- reading torch.save files ------------------------------------------------------------


class _StorageType:
    def __init__(self, name):
        self.dtype = _STORAGE_DTYPES[name]


class _Storage:
    def __init__(self, dtype, data):
        self.dtype, self.data = dtype, data


class _LazyTensor:
    def __init__(self, storage, offset, size, stride):
        self.storage, self.offset, self.size, self.stride = storage, offset, tuple(size), tuple(stride)


def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad=False, backward_hooks=None, metadata=None):
    return _LazyTensor(storage, storage_offset, size, stride)


def _rebuild_parameter(data, requires_grad=False, backward_hooks=None, *extra):
    return data


_ALLOWED = {
    ("collections", "OrderedDict"): collections.OrderedDict,
    ("torch._utils", "_rebuild_tensor_v2"): _rebuild_tensor_v2,
    ("torch._utils", "_rebuild_parameter"): _rebuild_parameter,
}


class _RestrictedUnpickler(pickle.Unpickler):
    def __init__(self, file, zf, prefix):
        super().__init__(file)
        self._zf, self._prefix = zf, prefix

    def find_class(self, module, name):
        if (module, name) in _ALLOWED:
            return _ALLOWED[(module, name)]
        if module == "torch" and name in _STORAGE_DTYPES:
            return _StorageType(name)
        raise pickle.UnpicklingError(
            f"refusing to load '{module}.{name}': only plain tensors and dicts are allowed in checkpoints"
        )

    def persistent_load(self, pid):
        if not (isinstance(pid, tuple) and pid and pid[0] == "storage"):
            raise pickle.UnpicklingError(f"unsupported persistent id {pid!r}")
        _, storage_type, key, _location, _numel = pid[:5]
        if not isinstance(storage_type, _StorageType):
            raise pickle.UnpicklingError("unsupported storage type")
        return _Storage(storage_type.dtype, self._zf.read(f"{self._prefix}data/{key}"))


def _contiguous(size, stride):
    expect = 1
    for dim, st in zip(reversed(size), reversed(stride)):
        if dim != 1 and st != expect:
            return False
        expect *= dim
    return True


def _materialize(obj, device):
    if isinstance(obj, _LazyTensor):
        if not _contiguous(obj.size, obj.stride):
            raise ValueError("checkpoint contains a non-contiguous tensor; save it with .contiguous()")
        itemsize = _ITEMSIZE[obj.storage.dtype]
        n = 1
        for d in obj.size:
            n *= d
        start = obj.offset * itemsize
        view = memoryview(obj.storage.data)[start : start + n * itemsize]
        return _native.tensor_from_buffer(view, obj.storage.dtype, list(obj.size), device)
    if isinstance(obj, dict):
        return type(obj)((k, _materialize(v, device)) for k, v in obj.items())
    return obj


def load_torch(path, device=None):
    """Load a checkpoint written by `torch.save` (the zip format) as a dict of tensors.

    Works without torch installed. Only tensors, dicts and plain values are supported; anything
    else in the pickle is rejected rather than executed.
    """
    with zipfile.ZipFile(path) as zf:
        pkl = next((n for n in zf.namelist() if n.endswith("data.pkl")), None)
        if pkl is None:
            raise ValueError("not a torch.save zip checkpoint (no data.pkl); legacy formats are unsupported")
        prefix = pkl[: -len("data.pkl")]
        obj = _RestrictedUnpickler(io.BytesIO(zf.read(pkl)), zf, prefix).load()
    return _materialize(obj, device)
