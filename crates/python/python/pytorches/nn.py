"""A minimal `nn`: Module, Linear, ReLU, Sequential, mse_loss (enough for MLPs).

`state_dict()` / `load_state_dict()` use PyTorch's names and layouts, so checkpoints move between
the two libraries: `Sequential` children are keyed by index (`"0.weight"`, `"2.bias"`), and
`Linear.weight` is exported as `[out_features, in_features]` like `torch.nn.Linear`.
"""

import math

from . import _native

_seed = [1000]


def _next_seed():
    _seed[0] += 1
    return _seed[0]


class Module:
    # ---- structure -------------------------------------------------------------------------

    def _children(self):
        """`(name, module)` pairs for sub-modules, in definition order."""
        for name, v in vars(self).items():
            if isinstance(v, Module):
                yield name, v
            elif isinstance(v, (list, tuple)):
                for i, m in enumerate(v):
                    if isinstance(m, Module):
                        yield f"{name}.{i}", m

    def _local_names(self):
        """Names of this module's own parameters, in PyTorch order."""
        return [k for k, v in vars(self).items() if isinstance(v, _native.Tensor) and v.requires_grad]

    # Hooks for modules whose in-memory layout differs from PyTorch's.
    def _export_local(self, name):
        return getattr(self, name).detach()

    def _import_local(self, name, value):
        param = getattr(self, name)
        if list(value.shape) != list(param.shape):
            raise ValueError(f"shape mismatch for '{name}': checkpoint {list(value.shape)}, model {list(param.shape)}")
        param.copy_(value)

    # ---- parameters ------------------------------------------------------------------------

    def parameters(self):
        out = [getattr(self, n) for n in self._local_names()]
        for _, child in self._children():
            out += child.parameters()
        return out

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    # ---- checkpoints -----------------------------------------------------------------------

    def state_keys(self, prefix=""):
        keys = [prefix + n for n in self._local_names()]
        for name, child in self._children():
            keys += child.state_keys(f"{prefix}{name}.")
        return keys

    def state_dict(self, prefix=""):
        """Parameters keyed by PyTorch-style names, in PyTorch's layouts (detached; copies where the
        layout differs, such as `Linear.weight`)."""
        out = {prefix + n: self._export_local(n) for n in self._local_names()}
        for name, child in self._children():
            out.update(child.state_dict(f"{prefix}{name}."))
        return out

    def load_state_dict(self, state, strict=True):
        """Copy tensors from `state` (e.g. from `torch.nn.Module.state_dict()` converted with
        `pytorches.from_torch_state_dict`, or `safetensors.load`) into this model's parameters."""
        expected = set(self.state_keys())
        missing = sorted(expected - set(state))
        unexpected = sorted(set(state) - expected)
        if strict and (missing or unexpected):
            raise KeyError(f"state_dict mismatch: missing {missing}, unexpected {unexpected}")
        self._load(state, "")
        return missing, unexpected

    def _load(self, state, prefix):
        for n in self._local_names():
            if prefix + n in state:
                self._import_local(n, state[prefix + n])
        for name, child in self._children():
            child._load(state, f"{prefix}{name}.")

    def __call__(self, *args):
        return self.forward(*args)


class Linear(Module):
    def __init__(self, in_features, out_features, device=None):
        self.in_features, self.out_features = in_features, out_features
        scale = math.sqrt(2.0 / in_features)  # He init: keeps activation variance stable through ReLU
        w = _native.randn([in_features, out_features], device, _next_seed()) * scale
        # Stored as [in, out] so forward is a plain matmul with no transposed copy.
        self.weight = w.detach().requires_grad_(True)
        self.bias = _native.full([out_features], 0.0, device, True)

    def forward(self, x):
        return x @ self.weight + self.bias

    def _export_local(self, name):
        w = getattr(self, name).detach()
        return w.t() if name == "weight" else w  # PyTorch layout: [out, in]

    def _import_local(self, name, value):
        if name == "weight":
            if list(value.shape) != [self.out_features, self.in_features]:
                raise ValueError(
                    f"shape mismatch for 'weight': checkpoint {list(value.shape)}, "
                    f"model {[self.out_features, self.in_features]}"
                )
            self.weight.copy_(value.t())
        else:
            super()._import_local(name, value)


class ReLU(Module):
    def forward(self, x):
        return x.relu()


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)

    def _children(self):
        # PyTorch names Sequential children by index, including parameter-free ones.
        for i, m in enumerate(self.layers):
            yield str(i), m

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def mlp(sizes, device=None):
    """`Linear -> ReLU` stack over `sizes`, e.g. `mlp([784, 256, 10])`."""
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(Linear(sizes[i], sizes[i + 1], device))
        if i < len(sizes) - 2:
            layers.append(ReLU())
    return Sequential(*layers)


def mse_loss(pred, target):
    d = pred - target
    return (d * d).mean()


def estimate_mlp_training_bytes(sizes, batch):
    """Approximate peak device memory (bytes) to train `mlp(sizes)` with batch size `batch`.

    Counts weights and their gradients, activations kept for backward, and the transient
    transposed-weight copies made during backward. An estimate, intentionally a bit generous.
    """
    params = sum(a * b + b for a, b in zip(sizes, sizes[1:]))
    acts = batch * sum(sizes)
    largest = max(a * b for a, b in zip(sizes, sizes[1:]))
    return 4 * (2 * params + 3 * acts + 2 * largest)
