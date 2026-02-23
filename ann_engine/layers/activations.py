import numpy as np
from ann_engine.core.tensor import Tensor
from ann_engine.layers.base import Module

class Relu(Module):
    """
    Rectified Linear Unit activatiob function
    f(x)= max(0,x)
    
    """
    def __init__(self):
        super().__init__()
        self.cache = None     # storing input for backward pass
        
    def forward(self,x):
        """
        Forward pass : Relu(x) = max(0,x)

        Args:
            x : Input Tensor
            
        Returns:
           Tensor with Relu applied
        """
        self.cache = x # storing input for backward pass
        out_data = np.maximum(0, x.data)
        out = Tensor(out_data, (x,),'ReLU')
        
        def _backward():
             # Gradient: 1 where x > 0, 0 elsewhere
            mask = (self.cache.data > 0).astype(np.float32)
            self.cache.grad += out.grad * mask
        out._backward = _backward
        return out
    
    def __repr__(self):
        return "ReLU"
    
    
class Sigmoid(Module):
    """
    Sigmoid Activation Function
    f(x) = 1/(1 + exp(-x))
    
    """
    def __init__(self):
        super().__init__()
        self.cache = None
        
    def forward(self, x):
        """
        Forward pass : Sigmoid(x) = 1/(1 - exp(-x))

        Args:
            x : Input Tensor
            
        Returns:
            Tensor with sigmoid applied
            
        """
        
        out_data = 1 /(1 + np.exp(-x.data))
        out = Tensor(out_data, (x,), "Sigmoid")
        self.cache = out_data
        
        def _backward():
             # Gradient: sigmoid(x) * (1 - sigmoid(x))
             
            grad = out.grad * (self.cache * (1-self.cache))
            x.grad += grad
        out._backward = _backward
        return out
    
    def __repr__(self):
        return "Sigmoid()"
    
    
class Tanh(Module):
    """
    Hyperbolic Tangent activation function
    f(x) = tanh(x)
    
    """
    
    def __init__(self):
        super().__init__()
        self.cache = None
        
    def forward(self,x):
        """
        forward pass : tanh(x)
        
        Args:
            X: Input Tensor
            
        Returns:
            Tensor with tanh applied
        """ 
        out_data = np.tanh(x.data)
        out = Tensor(out_data, (x,), 'Tanh')
        
        self.cache = out_data
        
        def _backward():
            
           # Gradient: 1 - tanh^2(x)
           grad = out.grad * (1-self.cache**2)
           x.grad += grad
        out._backward = _backward
        return out
    
    def __repr__(self):
        return "Tanh()"
    
    
    