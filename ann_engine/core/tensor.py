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
        saved_self = self
        def _backward():
            saved_self.grad += out.grad * np.ones_like(saved_self.data)
        out._backward = _backward
        return out
     
    def mean(self, axis=None, keepdims=False):
        if axis is None:
            out = Tensor(self.data.mean(), (self,), 'mean')
            saved_self = self
            def _backward():
                saved_self.grad += np.ones_like(saved_self.data) * (out.grad / saved_self.data.size)
            out._backward = _backward
            return out
        else:
            out = Tensor(self.data.mean(axis=axis, keepdims=keepdims), (self,), 'mean')
            saved_self = self
            saved_axis = axis
            saved_keepdims = keepdims
            def _backward():
                grad = out.grad / saved_self.data.shape[saved_axis]
                expand_shape = list(saved_self.data.shape)
                if not saved_keepdims:
                    expand_shape.insert(saved_axis if saved_axis >= 0 else len(expand_shape), 1)
                grad = grad.reshape(expand_shape)
                saved_self.grad += np.broadcast_to(grad, saved_self.data.shape)
            out._backward = _backward
            return out
    
    def reshape(self, *shape):
        out = Tensor(self.data.reshape(shape), (self,), 'reshape')
        saved_self = self
        def _backward():
            saved_self.grad += out.grad.reshape(saved_self.data.shape)
        out._backward = _backward
        return out
    
    @property
    def T(self):
        out = Tensor(self.data.T, (self,), 'T')
        saved_self = self
        def _backward():
            saved_self.grad += out.grad.T 
        out._backward = _backward
        return out

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')
        saved_self = self
        saved_other = other
        def _backward():
            if saved_self.data.shape != out.data.shape:
                axis_to_sum = []
                for i, (dim_self, dim_out) in enumerate(zip(saved_self.data.shape, out.data.shape)):
                    if dim_self != dim_out:
                        axis_to_sum.append(i)
                if axis_to_sum:
                    grad_self = out.grad.sum(axis=tuple(axis_to_sum), keepdims=True)
                    grad_self = grad_self.reshape(saved_self.data.shape)
                else:
                    grad_self = out.grad
            else:
                grad_self = out.grad

            if saved_other.data.shape != out.data.shape:
                axis_to_sum = []
                for i, (dim_other, dim_out) in enumerate(zip(saved_other.data.shape, out.data.shape)):
                    if dim_other != dim_out:
                        axis_to_sum.append(i)
                if axis_to_sum:
                    grad_other = out.grad.sum(axis=tuple(axis_to_sum), keepdims=True)
                    grad_other = grad_other.reshape(saved_other.data.shape)
                else:
                    grad_other = out.grad
            else:
                grad_other = out.grad

            saved_self.grad += grad_self
            saved_other.grad += grad_other
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, (self, other), '-')
        saved_self = self
        saved_other = other
        def _backward():
            if saved_self.data.shape != out.data.shape:
                axis = tuple(range(len(out.grad.shape) - len(saved_self.data.shape)))
                saved_self.grad += out.grad.sum(axis=axis)
            else:
                saved_self.grad += out.grad
                
            if saved_other.data.shape != out.data.shape:
                axis = tuple(range(len(out.grad.shape) - len(saved_other.data.shape)))
                saved_other.grad += (-out.grad).sum(axis=axis)
            else:
                saved_other.grad += -out.grad
        out._backward = _backward
        return out
    
    def __rsub__(self, other):
        return Tensor(other) - self

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')
        saved_self = self
        saved_other = other
        def _backward():
            grad_self = saved_other.data * out.grad
            grad_other = saved_self.data * out.grad
            
            if saved_self.data.shape != grad_self.shape:
                axis = tuple(range(len(grad_self.shape) - len(saved_self.data.shape)))
                saved_self.grad += grad_self.sum(axis=axis)
            else:
                saved_self.grad += grad_self
                
            if saved_other.data.shape != grad_other.shape:
                axis = tuple(range(len(grad_other.shape) - len(saved_other.data.shape)))
                saved_other.grad += grad_other.sum(axis=axis)
            else:
                saved_other.grad += grad_other
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, (self, other), '/')
        saved_self = self
        saved_other = other
        def _backward():
            grad_self = (1.0 / saved_other.data) * out.grad
            grad_other = (-saved_self.data / (saved_other.data ** 2)) * out.grad
            
            if saved_self.data.shape != grad_self.shape:
                axis = tuple(range(len(grad_self.shape) - len(saved_self.data.shape)))
                saved_self.grad += grad_self.sum(axis=axis)
            else:
                saved_self.grad += grad_self
                
            if saved_other.data.shape != grad_other.shape:
                axis = tuple(range(len(grad_other.shape) - len(saved_other.data.shape)))
                saved_other.grad += grad_other.sum(axis=axis)
            else:
                saved_other.grad += grad_other
        out._backward = _backward
        return out
    
    def __rtruediv__(self, other):
        return Tensor(other) / self
    
    def __pow__(self, power):
        assert isinstance(power, (int, float)), "Only supports int or float powers"
        out = Tensor(self.data ** power, (self,), f'**{power}')
        saved_self = self
        saved_power = power
        def _backward():
            saved_self.grad += (saved_power * saved_self.data ** (saved_power - 1)) * out.grad
        out._backward = _backward
        return out
    
    def __rpow__(self, other):
        raise NotImplementedError("Reverse power not implemented")
    
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), '@')
        saved_self = self
        saved_other = other
        def _backward():
            saved_self.grad += out.grad @ saved_other.data.T
            saved_other.grad += saved_self.data.T @ out.grad
        out._backward = _backward
        return out
    
    def __rmatmul__(self, other):
        return Tensor(other) @ self

    def __neg__(self):
        out = Tensor(-self.data, (self,), 'neg')
        saved_self = self
        def _backward():
            saved_self.grad += -out.grad
        out._backward = _backward
        return out

    def backward(self, retain_graph=False):
        topo = []
        visited = set()
        def build_topo(t):
            if id(t) not in visited:
                visited.add(id(t))
                for child in t._prev:
                    build_topo(child)
                topo.append(t)
        build_topo(self)

        if self.data.shape == ():
            self.grad = np.array(1.0, dtype=np.float32)
        else:
            self.grad = np.ones_like(self.data)

        for t in reversed(topo):
            t._backward()
            
        if not retain_graph:
            for t in topo:
                t._prev = set()

    def zero_grad(self):
        """Reset gradients to zero."""
        self.grad = np.zeros_like(self.data)

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"