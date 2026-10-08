"""A minimal `nn`: Module, Linear, ReLU, Sequential, mse_loss (enough for MLPs).

`state_dict()` / `load_state_dict()` use PyTorch's names and layouts, so checkpoints move between
the two libraries: `Sequential` children are keyed by index (`"0.weight"`, `"2.bias"`), and
`Linear.weight` is exported as `[out_features, in_features]` like `torch.nn.Linear`.
"""

import math

from . import _native

_seed = [1000]


def manual_seed(seed):
    """Make layers created after this call initialize identically on every run and placement."""
    _seed[0] = 1000 + seed


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

    # ---- placement -------------------------------------------------------------------------

    def to(self, device):
        """Move every parameter to `device`, in place on this module. Parameters become new leaf
        tensors (any `.grad` is dropped), so build the optimizer after moving."""
        device = _canonical(device)
        for n in self._local_names():
            p = getattr(self, n)
            if p.device != device:
                setattr(self, n, p.detach().to(device).requires_grad_(True))
        for _, child in self._children():
            child.to(device)
        return self

    def _cost(self, in_shape):
        """`(out_shape, forward_flops)` for an input of `in_shape`; lets the planner describe this
        module without running it. Modules that can sit in a placed chain implement this."""
        raise NotImplementedError(f"{type(self).__name__} cannot describe its cost to the planner")

    def __call__(self, *args):
        return self.forward(*args)


def _canonical(device):
    """`"cuda"` -> `"cuda:0"`, matching what `Tensor.device` reports."""
    return device if ":" in device else f"{device}:0"


def _prod(shape):
    n = 1
    for s in shape:
        n *= s
    return n


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

    def _cost(self, in_shape):
        if in_shape[-1] != self.in_features:
            raise ValueError(f"Linear expects {self.in_features} input features, got shape {list(in_shape)}")
        rows = _prod(in_shape[:-1])
        return list(in_shape[:-1]) + [self.out_features], 2 * rows * self.in_features * self.out_features

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

    def _cost(self, in_shape):
        return list(in_shape), _prod(in_shape)


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)
        self.layer_devices = None  # set by place(); one device per layer

    def _children(self):
        # PyTorch names Sequential children by index, including parameter-free ones.
        for i, m in enumerate(self.layers):
            yield str(i), m

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if self.layer_devices is not None and x.device != self.layer_devices[i]:
                x = x.to(self.layer_devices[i])  # differentiable: the gradient comes back
            x = layer(x)
        return x

    def _cost(self, in_shape):
        flops, shape = 0, list(in_shape)
        for layer in self.layers:
            shape, f = layer._cost(shape)
            flops += f
        return shape, flops

    def graph_info(self, input_shape):
        """Describe this chain to the planner for an input of `input_shape` (a `GraphInfo`)."""
        shape, layers = list(input_shape), []
        for i, layer in enumerate(self.layers):
            shape, flops = layer._cost(shape)
            layers.append((f"{i}:{type(layer).__name__}", 4 * layer.num_parameters(), 4 * _prod(shape), flops))
        return _native.GraphInfo(4 * _prod(input_shape), layers)

    def to(self, device):
        super().to(device)
        self.layer_devices = [_canonical(device)] * len(self.layers)
        return self

    def place(self, devices):
        """Run each layer on its own device: `devices` has one entry per layer, or is a single
        device for all, or a `ModelPlan` from `pytorches.plan_model`. Moves the parameters now, and
        moves activations between devices in `forward` (and their gradients in `backward`). Inputs
        may be anywhere; the output is on `output_device`, so put targets there. Build the
        optimizer after calling this."""
        devices = getattr(devices, "devices", devices)
        if isinstance(devices, str):
            devices = [devices] * len(self.layers)
        devices = [_canonical(d) for d in devices]
        if len(devices) != len(self.layers):
            raise ValueError(f"got {len(devices)} device(s) for {len(self.layers)} layer(s)")
        for layer, dev in zip(self.layers, devices):
            layer.to(dev)
        self.layer_devices = devices
        return self

    @property
    def input_device(self):
        return self.layer_devices[0] if self.layer_devices else None

    @property
    def output_device(self):
        return self.layer_devices[-1] if self.layer_devices else None


def mlp(sizes, device=None):
    """`Linear -> ReLU` stack over `sizes`, e.g. `mlp([784, 256, 10])`.

    `device` is one device for the whole model, or a placement (a list with one device per layer,
    ReLUs included, or a `ModelPlan`). With a placement every layer is created directly on its
    device, so a model too big for any one device never has to exist in one place."""
    devices = getattr(device, "devices", device)
    n_layers = 2 * (len(sizes) - 1) - 1
    per_layer = devices if isinstance(devices, (list, tuple)) else [devices] * n_layers
    if len(per_layer) != n_layers:
        raise ValueError(f"got {len(per_layer)} device(s) for the {n_layers} layers of mlp({list(sizes)})")
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(Linear(sizes[i], sizes[i + 1], per_layer[len(layers)]))
        if i < len(sizes) - 2:
            layers.append(ReLU())
    model = Sequential(*layers)
    if isinstance(devices, (list, tuple)):
        model.layer_devices = [_canonical(d) for d in devices]
    return model


def mlp_graph(sizes, batch):
    """The planner's description of `mlp(sizes)` at `batch`, without creating any tensors."""
    layers, shape = [], [batch, sizes[0]]
    for i in range(len(sizes) - 1):
        lin = (sizes[i], sizes[i + 1])
        out = [batch, lin[1]]
        layers.append((f"{len(layers)}:Linear", 4 * (lin[0] * lin[1] + lin[1]), 4 * _prod(out), 2 * batch * lin[0] * lin[1]))
        shape = out
        if i < len(sizes) - 2:
            layers.append((f"{len(layers)}:ReLU", 0, 4 * _prod(shape), _prod(shape)))
    return _native.GraphInfo(4 * batch * sizes[0], layers)


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
