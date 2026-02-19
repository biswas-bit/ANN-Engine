import numpy as np
from ann_engine.optimizers.base import Optimizer

class SGD(Optimizer):
    def __init__(self, parameters, lr=0.01):
        super().__init__(parameters, lr)
        
    def step(self):
        for param in self.parameters:
            if param.grad is not None:
                if param.grad.shape != param.data.shape:
                    if param.grad.shape == () and param.data.shape != ():
                        param.grad = np.ones_like(param.data) * param.grad
                    else:
                        raise ValueError(f"Gradient shape {param.grad.shape} doesn't match data shape {param.data.shape}")
                param.data -= self.lr * param.grad
                    
    def zero_grad(self):
        for param in self.parameters:
          
            param.grad = np.zeros_like(param.data)