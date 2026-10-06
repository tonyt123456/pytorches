"""A minimal `nn`: Module, Linear, ReLU, Sequential, mse_loss (enough for MLPs)."""

import math

from . import _native

_seed = [1000]


def _next_seed():
    _seed[0] += 1
    return _seed[0]


class Module:
    def parameters(self):
        out = []
        for v in vars(self).values():
            if isinstance(v, _native.Tensor) and v.requires_grad:
                out.append(v)
            elif isinstance(v, Module):
                out += v.parameters()
            elif isinstance(v, (list, tuple)):
                for m in v:
                    if isinstance(m, Module):
                        out += m.parameters()
        return out

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def __call__(self, *args):
        return self.forward(*args)


class Linear(Module):
    def __init__(self, in_features, out_features, device=None):
        self.in_features, self.out_features = in_features, out_features
        scale = math.sqrt(2.0 / in_features)  # He init: keeps activation variance stable through ReLU
        w = _native.randn([in_features, out_features], device, _next_seed()) * scale
        self.weight = w.detach().requires_grad_(True)
        self.bias = _native.full([out_features], 0.0, device, True)

    def forward(self, x):
        return x @ self.weight + self.bias


class ReLU(Module):
    def forward(self, x):
        return x.relu()


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)

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
