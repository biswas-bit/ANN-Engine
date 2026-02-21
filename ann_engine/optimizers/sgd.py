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
            
    def state_dict(self):
        """Return optimizer state for saving/loading"""
        return {
            'hyperparameters': {
                'lr': self.lr
            }
        }
    
    def load_state_dict(self, state_dict):
        """Load optimizer state from a state_dict"""
        hyperparams = state_dict['hyperparameters']
        self.lr = hyperparams['lr']
            
            
class SGDWithMomentum(SGD):
    def __init__(self, parameters, lr=0.01, momentum=0.9):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.velocities = [np.zeros_like(param.data) for param in parameters]
        
    def step(self):
        for i, param in enumerate(self.parameters):
            if param.grad is not None:
                grad = param.grad
                # handle scalar broadcast
                if grad.shape != param.data.shape:
                    if grad.shape == () and param.data.shape != ():
                        grad = np.ones_like(param.data) * grad
                    else:
                        raise ValueError(f"Gradient shape {grad.shape} doesn't match data shape {param.data.shape}")
                # update velocity
                self.velocities[i] = self.momentum * self.velocities[i] + self.lr * grad
                param.data -= self.velocities[i]

    def zero_grad(self):
        super().zero_grad()