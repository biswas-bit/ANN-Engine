import numpy as np
from .base import Optimizer

class NAG(Optimizer):
    def __init__(self, parameters, lr=0.01, momentum=0.9):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.velocities = [np.zeros_like(params.data) for params in self.parameters]
    
    def step(self):
        for i, param in enumerate(self.parameters):
            if param.grad is not None:
                continue
            
            if param.grad.shape != param.data.shape:
                raise ValueError(f"Gradient shape {param.grad.shape} doesn't match data shape {param.data.shape}")
            
            self.velocities[i] = self.momentum* self.velocities[i] + self.lr*param.grad
            #nesterov update
            param.data -= self.lr*(param.grad + self.momentum*self.velocities[i])
    
    def zero_grad(self):
        super().zero_grad()
        
    
    def state_dict(self):
        """Return optimizer state for saving/loading"""
        return {
            'velocities': self.velocities,
            'hyperparameters': {
                'lr': self.lr,
                'momentum': self.momentum
            }
        }
        
    def load_state_dict(self, state_dict):
        """Load optimizer state from a state_dict"""
        self.velocities = state_dict['velocities']
        hyperparams = state_dict['hyperparameters']
        self.lr = hyperparams['lr']
        self.momentum = hyperparams['momentum']