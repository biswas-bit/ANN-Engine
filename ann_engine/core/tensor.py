import numpy as np


class Tensor:
    def __init__(self, data, _children=(), _op=''):
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data)
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    # ------------------------------------------------------------------
    # Reduction ops
    # ------------------------------------------------------------------

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
                saved_self.grad += (
                    np.ones_like(saved_self.data) * (out.grad / saved_self.data.size)
                )
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
                    insert_at = saved_axis if saved_axis >= 0 else len(expand_shape)
                    expand_shape.insert(insert_at, 1)
                grad = grad.reshape(expand_shape)
                saved_self.grad += np.broadcast_to(grad, saved_self.data.shape)
            out._backward = _backward
            return out

    # ------------------------------------------------------------------
    # Shape ops
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Arithmetic ops
    # ------------------------------------------------------------------

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')
        saved_self = self
        saved_other = other

        def _backward():
            # Handle broadcast: sum over axes that were broadcast
            def _reduce_grad(grad, target_shape):
                # Sum over leading dims if grad has more dims
                while grad.ndim > len(target_shape):
                    grad = grad.sum(axis=0)
                # Sum over axes where target has size 1
                for i, dim in enumerate(target_shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)
                return grad

            saved_self.grad  += _reduce_grad(out.grad, saved_self.data.shape)
            saved_other.grad += _reduce_grad(out.grad, saved_other.data.shape)

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
            def _reduce_grad(grad, target_shape):
                while grad.ndim > len(target_shape):
                    grad = grad.sum(axis=0)
                for i, dim in enumerate(target_shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)
                return grad

            saved_self.grad  += _reduce_grad(out.grad, saved_self.data.shape)
            saved_other.grad -= _reduce_grad(out.grad, saved_other.data.shape)

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
            grad_self  = saved_other.data * out.grad
            grad_other = saved_self.data  * out.grad

            def _reduce(grad, target_shape):
                while grad.ndim > len(target_shape):
                    grad = grad.sum(axis=0)
                for i, dim in enumerate(target_shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)
                return grad

            saved_self.grad  += _reduce(grad_self,  saved_self.data.shape)
            saved_other.grad += _reduce(grad_other, saved_other.data.shape)

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
            grad_self  = (1.0 / saved_other.data) * out.grad
            grad_other = (-saved_self.data / (saved_other.data ** 2)) * out.grad

            def _reduce(grad, target_shape):
                while grad.ndim > len(target_shape):
                    grad = grad.sum(axis=0)
                for i, dim in enumerate(target_shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)
                return grad

            saved_self.grad  += _reduce(grad_self,  saved_self.data.shape)
            saved_other.grad += _reduce(grad_other, saved_other.data.shape)

        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        return Tensor(other) / self

    def __pow__(self, power):
        assert isinstance(power, (int, float)), "Only int/float powers supported"
        out = Tensor(self.data ** power, (self,), f'**{power}')
        saved_self  = self
        saved_power = power
        def _backward():
            saved_self.grad += (
                saved_power * saved_self.data ** (saved_power - 1)
            ) * out.grad
        out._backward = _backward
        return out

    def __rpow__(self, other):
        raise NotImplementedError("Reverse power not implemented")

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), '@')
        saved_self  = self
        saved_other = other
        def _backward():
            saved_self.grad  += out.grad @ saved_other.data.T
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

    # ------------------------------------------------------------------
    # Math ops
    # ------------------------------------------------------------------

    def log(self):
        out = Tensor(np.log(self.data + 1e-8), (self,), 'log')   # +eps for stability
        saved_self = self
        def _backward():
            saved_self.grad += out.grad / (saved_self.data + 1e-8)
        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), 'exp')
        saved_self = self
        def _backward():
            saved_self.grad += out.grad * out.data
        out._backward = _backward
        return out

    def sqrt(self):
        out = Tensor(np.sqrt(self.data), (self,), 'sqrt')
        saved_self = self
        def _backward():
            saved_self.grad += out.grad * (0.5 / (out.data + 1e-8))
        out._backward = _backward
        return out

    def square(self):
        return self ** 2

    def clip(self, min_val=None, max_val=None):
        """Clip values — NOTE: gradients do not flow through clipped regions."""
        clipped = np.clip(self.data, min_val, max_val)
        out = Tensor(clipped, (self,), 'clip')
        saved_self = self
        def _backward():
            # Pass gradient only where values were not clipped
            mask = np.ones_like(saved_self.data)
            if min_val is not None:
                mask[saved_self.data < min_val] = 0.0
            if max_val is not None:
                mask[saved_self.data > max_val] = 0.0
            saved_self.grad += out.grad * mask
        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Comparison ops (no gradient — used for masking only)
    # ------------------------------------------------------------------

    def __gt__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self.data > other.data

    def __lt__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self.data < other.data

    # ------------------------------------------------------------------
    # Backprop
    # ------------------------------------------------------------------

    def backward(self, retain_graph=False):
        """
        Reverse-mode autodiff via topological sort.

        Args:
            retain_graph: If True, keep the computation graph so backward()
                          can be called again (e.g. for debugging).
                          Default False — but we no longer destroy _prev
                          to avoid breaking subsequent forward passes.
        """
        # Build topological order
        topo    = []
        visited = set()

        def build_topo(t):
            if id(t) not in visited:
                visited.add(id(t))
                for child in t._prev:
                    build_topo(child)
                topo.append(t)

        build_topo(self)

        # Seed gradient
        if self.data.shape == ():
            self.grad = np.array(1.0, dtype=np.float32)
        else:
            self.grad = np.ones_like(self.data)

        # Propagate gradients
        for t in reversed(topo):
            t._backward()

        # ── KEY FIX ──────────────────────────────────────────────────
        # Do NOT clear _prev here.  Clearing _prev destroys the graph so
        # the next forward pass produces Tensors whose ancestors are gone,
        # making all subsequent backward() calls a no-op (zero gradients).
        #
        # Gradient accumulation is handled by zero_grad() in the optimizer,
        # which resets .grad before each backward pass — that is correct and
        # sufficient.  We never need to wipe _prev.
        # ─────────────────────────────────────────────────────────────

    def zero_grad(self):
        """Reset this tensor's gradient to zero."""
        self.grad = np.zeros_like(self.data)

    def __abs__(self):
        out = Tensor(np.abs(self.data), (self,), 'abs')
        saved_self = self
        def _backward():
            saved_self.grad += out.grad * np.sign(saved_self.data)
        out._backward = _backward
        return out

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"