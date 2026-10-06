"""Optimizers. Updates are in place, so parameters keep their identity."""


class SGD:
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self):
        for p in self.params:
            if p.grad is not None:
                p.copy_(p.detach() - p.grad * self.lr)
