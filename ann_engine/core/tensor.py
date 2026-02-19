import numpy as np

class Tensor:
    def __init__(self, data, _children=(), _op=''):
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data)
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
    
    def reshape(self, *shape):
        out = Tensor(self.data.reshape(shape), (self,), 'reshape')
        
        def _backward():
            self.grad += out.grad.reshape(self.data.shape)
            
        out._backward = _backward
        return out
    
    @property
    def T(self):
        out = Tensor(self.data.T, (self,), 'T')
        
        def _backward():
            self.grad += out.grad.T 
        
        out._backward = _backward
        return out

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')
        
        def _backward():
            # Handle broadcasting for self
            if self.data.shape != out.data.shape:
                # Sum over broadcasted dimensions
                axis = tuple(range(len(out.grad.shape) - len(self.data.shape)))
                self.grad += out.grad.sum(axis=axis)
            else:
                self.grad += out.grad
                
            # Handle broadcasting for other
            if other.data.shape != out.data.shape:
                axis = tuple(range(len(out.grad.shape) - len(other.data.shape)))
                other.grad += out.grad.sum(axis=axis)
            else:
                other.grad += out.grad
            
        out._backward = _backward
        return out
    
    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, (self, other), '-')
        def _backward():
            # Handle broadcasting for self
            if self.data.shape != out.data.shape:
                axis = tuple(range(len(out.grad.shape) - len(self.data.shape)))
                self.grad += out.grad.sum(axis=axis)
            else:
                self.grad += out.grad
                
            # Handle broadcasting for other
            if other.data.shape != out.data.shape:
                axis = tuple(range(len(out.grad.shape) - len(other.data.shape)))
                other.grad += (-out.grad).sum(axis=axis)
            else:
                other.grad += -out.grad
            
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')
        def _backward():
            # Calculate gradients
            grad_self = other.data * out.grad
            grad_other = self.data * out.grad
            
            # Handle broadcasting for self
            if self.data.shape != grad_self.shape:
                axis = tuple(range(len(grad_self.shape) - len(self.data.shape)))
                self.grad += grad_self.sum(axis=axis)
            else:
                self.grad += grad_self
                
            # Handle broadcasting for other
            if other.data.shape != grad_other.shape:
                axis = tuple(range(len(grad_other.shape) - len(other.data.shape)))
                other.grad += grad_other.sum(axis=axis)
            else:
                other.grad += grad_other
            
        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, (self, other), '/')
        def _backward():
            # Calculate gradients
            grad_self = (1.0 / other.data) * out.grad
            grad_other = (-self.data / (other.data ** 2)) * out.grad
            
            # Handle broadcasting for self
            if self.data.shape != grad_self.shape:
                axis = tuple(range(len(grad_self.shape) - len(self.data.shape)))
                self.grad += grad_self.sum(axis=axis)
            else:
                self.grad += grad_self
                
            # Handle broadcasting for other
            if other.data.shape != grad_other.shape:
                axis = tuple(range(len(grad_other.shape) - len(other.data.shape)))
                other.grad += grad_other.sum(axis=axis)
            else:
                other.grad += grad_other
            
        out._backward = _backward
        return out
    
    def __pow__(self, power):
        assert isinstance(power, (int, float)), "Only supports int or float powers"
        out = Tensor(self.data ** power, (self,), f'**{power}')
        def _backward():
            self.grad += (power * self.data ** (power - 1)) * out.grad
            
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

        # Initialize gradient of output correctly
        if self.data.shape == ():  # Scalar output
            self.grad = np.array(1.0, dtype=np.float32)
        else:  # Non-scalar output
            self.grad = np.ones_like(self.data)

        # Traverse nodes in reverse topological order
        for t in reversed(topo):
            t._backward()

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"


