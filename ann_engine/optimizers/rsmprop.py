import numpy as np
from ann_engine.optimizers.base import Optimizer

class RMSProp(Optimizer):
    def __init__(self, parameters, lr=0.01, beta=0.9, epsilon=1e-5):
        super().__init__(parameters, lr)
        self.beta = beta
        self.epsilon = epsilon
        self.V = [np.zeros_like(param.data) for param in self.parameters]
        
    def step(self):
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            if param.grad.shape != param.data.shape:
                raise ValueError(
                    f"Gradient shape {param.grad.shape} doesn't match data shape {param.data.shape}"
                )
            
            # Update moving average of squared gradients
            self.V[i] = self.beta * self.V[i] + (1 - self.beta) * param.grad ** 2
            
            # Parameter update
            param.data -= self.lr * param.grad / (np.sqrt(self.V[i] + self.epsilon))
            
    def zero_grad(self):
        super().zero_grad()