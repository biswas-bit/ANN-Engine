import numpy as np
from .base import Optimizer

class Adam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Adam optimizer
        
        Args:
            lr: Learning rate
            beta1: Exponential decay rate for first moment estimates
            beta2: Exponential decay rate for second moment estimates
            epsilon: Small constant for numerical stability
        """
        super().__init__(lr)  # Base class now expects lr only
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.parameters = None
        self.m = []
        self.v = []
        self.t = 0
    
    def set_parameters(self, parameters):
        """Set the parameters to optimize (called by model.compile)"""
        self.parameters = parameters
        self._initialize_moments()
    
    def _initialize_moments(self):
        """Initialize moment arrays for all parameters"""
        self.m = []
        self.v = []
        for param in self.parameters:
            self.m.append(np.zeros_like(param.data, dtype=np.float32))
            self.v.append(np.zeros_like(param.data, dtype=np.float32))
    
    def step(self):
        """Perform a single optimization step"""
        if self.parameters is None:
            raise RuntimeError("Adam optimizer parameters not set. Call set_parameters() first.")
        
        self.t += 1
        bias_correction1 = 1 - self.beta1 ** self.t
        bias_correction2 = 1 - self.beta2 ** self.t
        
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            # Ensure grad is float32 and has correct shape
            grad = param.grad.astype(np.float32)
            
            # Handle shape mismatches (e.g., scalar grad for non-scalar param)
            if grad.shape != param.data.shape:
                if grad.shape == () and param.data.shape != ():
                    grad = np.ones_like(param.data) * grad
                else:
                    raise ValueError(
                        f"Gradient shape {grad.shape} does not match "
                        f"parameter shape {param.data.shape}"
                    )
            
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
        if self.parameters is None:
            return
        for param in self.parameters:
            param.grad = np.zeros_like(param.data)
    
    def state_dict(self):
        """Return optimizer state for saving/loading"""
        if self.parameters is None:
            return {
                'm': [],
                'v': [],
                't': self.t,
                'hyperparameters': {
                    'lr': self.lr,
                    'beta1': self.beta1,
                    'beta2': self.beta2,
                    'epsilon': self.epsilon
                }
            }
        
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
        """Load optimizer state"""
        self.m = [m_i.copy() for m_i in state_dict['m']]
        self.v = [v_i.copy() for v_i in state_dict['v']]
        self.t = state_dict['t']
        
        if 'hyperparameters' in state_dict:
            hyperparams = state_dict['hyperparameters']
            self.lr = hyperparams.get('lr', self.lr)
            self.beta1 = hyperparams.get('beta1', self.beta1)
            self.beta2 = hyperparams.get('beta2', self.beta2)
            self.epsilon = hyperparams.get('epsilon', self.epsilon)
    
    def __repr__(self):
        return (f"Adam(lr={self.lr}, beta1={self.beta1}, "
                f"beta2={self.beta2}, epsilon={self.epsilon})")