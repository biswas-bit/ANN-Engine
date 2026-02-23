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
            mask = (self.cache.data > 0).astype(np.float32)
            self.cache.grad += out.grad * mask
        out._backward = _backward
        return out
    
    def __repr__(self):
        return "ReLU"
    
    