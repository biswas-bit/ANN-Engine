import numpy as np
from ann_engine.optimizers.base import Optimizer

class AdaGrad(Optimizer):
    def __init__(self, parameters, lr=0.01, epsilon=1e-8):
        super().__init__(parameters, lr)
        self.epsilon = epsilon
        self.G = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue

            # Ensure gradient shape matches parameter shape
            if param.grad.shape != param.data.shape:
                raise ValueError(
                    f"Gradient shape {param.grad.shape} "
                    f"doesn't match data shape {param.data.shape}"
                )

            # Accumulate squared gradient
            self.G[i] += param.grad ** 2

            # Parameter update
            param.data -= self.lr * param.grad / (np.sqrt(self.G[i]) + self.epsilon)

    def zero_grad(self):
        super().zero_grad()