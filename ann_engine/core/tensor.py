import numpy as np

class Tensor:
    def __init__(self, data, _children=(), _op=''):
        self.data = np.array(data, dtype=np.float32)
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None  

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        # backward function
        def _backward():
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0
        out._backward = _backward
        return out
    
    def __sub__(self, other):
        other = other if isinstance(other,Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, (self, other), '-')
        
        def _backward():
            self.grad +=out.grad *1.0
            other.grad += out.grad*(-1.0)
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        other = other if isinstance(other, other) else Tensor(other)
        out = Tensor(self.data / other.data, (self, other), '/')
        return out

    def backward(self):
        # Topological sort of nodes to handle dependencies
        topo = []
        visited = set()
        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for child in t._prev:
                    build_topo(child)
                topo.append(t)
        build_topo(self)

        # Initialize gradient of output
        self.grad = 1.0

        # Traverse nodes in reverse topological order
        for t in reversed(topo):
            t._backward()

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"