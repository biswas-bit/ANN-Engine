import numpy as np
from ann_engine.optimizers.base import Optimizer

class Adam(Optimizer):
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        # Note: using 'lr' instead of 'learning_rate' to match base class
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = []
        self.v = []
        self.t = 0
        
        # Initialize moments for each parameter
        for param in self.parameters:
            self.m.append(np.zeros_like(param.data, dtype=np.float32))
            self.v.append(np.zeros_like(param.data, dtype=np.float32))
        
    def step(self):
        self.t += 1
        bias_correction1 = 1 - self.beta1 ** self.t
        bias_correction2 = 1 - self.beta2 ** self.t
        
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            # Ensure grad is float32 and has correct shape
            grad = param.grad.astype(np.float32)
            
            # Handle shape mismatches
            if grad.shape != param.data.shape:
                if grad.shape == () and param.data.shape != ():
                    grad = np.ones_like(param.data) * grad
                else:
                    raise ValueError(f"Gradient shape {grad.shape} does not match parameter shape {param.data.shape}")
            
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Update biased second moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected estimates
            m_hat = self.m[i] / bias_correction1
            v_hat = self.v[i] / bias_correction2
            
            # Parameter update (using self.lr from base class)
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
    
    def zero_grad(self):
        """Reset gradients to zero"""
        for param in self.parameters:
            param.grad = np.zeros_like(param.data)