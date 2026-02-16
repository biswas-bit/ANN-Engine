import numpy as np

class Tensor:
    def __init__(self, data, _children=(), _op=""):
        self.data = np.array(data, dtype=np.float32)
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
    
    def __add__(self, other):
        other = other if isinstance(other,Tensor) else Tensor(other)
        return Tensor(self.data+other.data, (self, other), '+')
    
    def __mul__(self, other):
        other = other if isinstance(other,Tensor) else Tensor(other)
        return Tensor(self.data*other.data, (self, other),"*")
    
    def __repr__(self):
        return f"Tensor(data={self.data})"