import numpy as np
from .base import Optimizer

class AdaGrad(Optimizer):
    def __init__(self, parameters, lr=0.01, epsilon=1e-8, weight_decay=0, clip_grad=None):
        super().__init__(parameters, lr)
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.clip_grad = clip_grad
        self.G = []
        
        # Initialize squared gradient accumulator for each parameter
        for param in self.parameters:
            self.G.append(np.zeros_like(param.data, dtype=np.float32))

    def step(self):
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue

            # Ensure grad is float32
            grad = param.grad.astype(np.float32)

            # Handle shape mismatches (scalar grad for non-scalar param)
            if grad.shape != param.data.shape:
                if grad.shape == () and param.data.shape != ():
                    grad = np.ones_like(param.data) * grad
                else:
                    raise ValueError(
                        f"Gradient shape {grad.shape} "
                        f"doesn't match data shape {param.data.shape}"
                    )

            # Apply weight decay (L2 regularization)
            if self.weight_decay != 0:
                grad += self.weight_decay * param.data

            # Gradient clipping (optional)
            if self.clip_grad is not None:
                grad = np.clip(grad, -self.clip_grad, self.clip_grad)

            # Accumulate squared gradient
            self.G[i] += grad ** 2

            # Parameter update with numerical stability
            param.data -= self.lr * grad / (np.sqrt(self.G[i]) + self.epsilon)

    def zero_grad(self):
        """Reset gradients to zero"""
        for param in self.parameters:
            param.grad = np.zeros_like(param.data)
    
    def state_dict(self):
        """Return optimizer state for saving/loading"""
        return {
            'G': self.G,
            'hyperparameters': {
                'lr': self.lr,
                'epsilon': self.epsilon,
                'weight_decay': self.weight_decay,
                'clip_grad': self.clip_grad
            }
        }
    
    def load_state_dict(self, state_dict):
        """Load optimizer state"""
        self.G = state_dict['G']
        if 'hyperparameters' in state_dict:
            hp = state_dict['hyperparameters']
            self.lr = hp.get('lr', self.lr)
            self.epsilon = hp.get('epsilon', self.epsilon)
            self.weight_decay = hp.get('weight_decay', self.weight_decay)
            self.clip_grad = hp.get('clip_grad', self.clip_grad)