import numpy as np

class Tensor:
    def __init__(self, data, _children=(), _op=''):
        self.data = np.array(data, dtype=np.float32)
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None 
        
    def sum(self):
        out = Tensor(self.data.sum(), (self,), 'sum')
        def _backward():
            self.grad += out.grad * np.ones_like(self.data)
            
        out._backward = _backward
        return out
     
     
    def mean(self):
        n_elements = self.data.size
        out = Tensor(self.data.mean(), (self,), 'mean')
        def _backward():
            self.grad += np.ones_like(self.data) * (out.grad / n_elements)
            
        out._backward = _backward
        return out

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')
        
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
        def _backward():
            self.grad +=(1.0/other.data)*out.grad
            other.grad +=(-self.data/(other.data**2))*out.grad
            
        out._backward = _backward
        return out
    
    def __pow__(self, power):
        assert isinstance(power,(int, float)), "Only supports int or float powers"
        out = Tensor(self.data ** power, (self,), f'**{power}')
        def _backward():
            self.grad += (power * self.data ** (power -1))* out.grad
            
        out._backward = _backward
        return out
    
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), '@')
        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
            
        out._backward = _backward
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