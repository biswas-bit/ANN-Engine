import numpy as np
from ann_engine.optimizers.base import Optimizer
import copy

class Adam(Optimizer):
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
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
            
            # Parameter update 
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
    
    def zero_grad(self):
        """Reset gradients to zero"""
        for param in self.parameters:
            param.grad = np.zeros_like(param.data)
            
    def state_dict(self):
        """ Return optimizer state for saving/loading """
        return {
            'm': [m_i.copy() for m_i in self.m], 
            'v': [v_i.copy() for v_i in self.v],  
            't': self.t,
            'hyperparameters': {
                'lr': self.lr,
                'beta1': self.beta1,
                'beta2': self.beta2,
                'epsilon': self.epsilon
            }
        }
        
    def load_state_dict(self, state_dict):
        """ Load optimizer state """
        self.m = [m_i.copy() for m_i in state_dict['m']]  
        self.v = [v_i.copy() for v_i in state_dict['v']]  
        self.t = state_dict['t']
        
        if 'hyperparameters' in state_dict:
            hyperparams = state_dict['hyperparameters']
            self.lr = hyperparams['lr']
            self.beta1 = hyperparams['beta1']
            self.beta2 = hyperparams['beta2']
            self.epsilon = hyperparams['epsilon']