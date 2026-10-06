"""Read and write the safetensors format (https://huggingface.co/docs/safetensors).

Files are memory-mapped and each tensor is copied straight to its device. Tensors of any
supported dtype (F32, F16, BF16, F64, I8..I64, U8..U64, BOOL) load as float32, which is the
only dtype PyTorches computes in today. `save` writes float32.
"""

import json
import mmap
import struct

from . import _native

_MAX_HEADER = 100 * 1024 * 1024  # same sanity limit as the reference implementation


def _read_header(f):
    raw = f.read(8)
    if len(raw) < 8:
        raise ValueError("not a safetensors file: too short")
    (n,) = struct.unpack("<Q", raw)
    if n > _MAX_HEADER:
        raise ValueError(f"safetensors header too large ({n} bytes)")
    body = f.read(n)
    if len(body) < n:
        raise ValueError("safetensors header is truncated")
    try:
        header = json.loads(body)
    except json.JSONDecodeError as e:
        raise ValueError(f"safetensors header is not valid JSON: {e}") from None
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be a JSON object")
    return header, 8 + n


def metadata(path):
    """The free-form string metadata stored in the file (empty dict if none)."""
    with open(path, "rb") as f:
        header, _ = _read_header(f)
    return dict(header.get("__metadata__", {}))


def load(path, device=None):
    """Load every tensor in `path` into a dict of `pytorches.Tensor` on `device`."""
    out = {}
    with open(path, "rb") as f:
        header, base = _read_header(f)
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            size = len(mm) - base
            with memoryview(mm) as view:
                for name, info in header.items():
                    if name == "__metadata__":
                        continue
                    start, end = info["data_offsets"]
                    if not 0 <= start <= end <= size:
                        raise ValueError(f"tensor '{name}' has out-of-range data offsets {start}..{end}")
                    out[name] = _native.tensor_from_buffer(
                        view[base + start : base + end], info["dtype"], list(info["shape"]), device
                    )
        finally:
            mm.close()
    return out


def save(tensors, path, metadata=None):
    """Write a dict of name -> `pytorches.Tensor` as a float32 safetensors file."""
    header, chunks, offset = {}, [], 0
    for name in sorted(tensors):
        raw = _native.tensor_to_bytes(tensors[name])
        header[name] = {
            "dtype": "F32",
            "shape": list(tensors[name].shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        chunks.append(raw)
        offset += len(raw)
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    body = json.dumps(header, separators=(",", ":")).encode()
    body += b" " * (-len(body) % 8)  # align the data section to 8 bytes
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(body)))
        f.write(body)
        for raw in chunks:
            f.write(raw)
