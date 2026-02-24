import numpy as np
from ann_engine.core.tensor import Tensor
from ann_engine.layers.base import Module

class ReLU(Module):
    """
    Rectified Linear Unit activation function
    f(x) = max(0, x)
    """
    
    def __init__(self):
        super().__init__()
        # Store cache for each forward pass
        self.cache = {}
        self.forward_count = 0
    
    def forward(self, x):
        """
        Forward pass: ReLU(x) = max(0, x)
        
        Args:
            x: Input Tensor
        
        Returns:
            Tensor with ReLU applied
        """
        # Store input for this forward pass
        cache_id = self.forward_count
        self.forward_count += 1
        
        # Apply ReLU: max(0, x)
        out_data = np.maximum(0, x.data)
        out = Tensor(out_data, (x,), 'ReLU')
        
        def _backward(cache_id=cache_id):
            # Get the input for this specific forward pass
            # Gradient: 1 where x > 0, 0 elsewhere
            mask = (x.data > 0).astype(np.float32)
            x.grad += out.grad * mask
        
        out._backward = _backward
        return out
    
    def __repr__(self):
        return "ReLU()"


class Sigmoid(Module):
    """
    Sigmoid activation function
    f(x) = 1 / (1 + exp(-x))
    """
    
    def __init__(self):
        super().__init__()
        self.cache = None  # Store output for backward pass
    
    def forward(self, x):
        """
        Forward pass: sigmoid(x) = 1 / (1 + exp(-x))
        
        Args:
            x: Input Tensor
        
        Returns:
            Tensor with sigmoid applied
        """
        # Compute sigmoid
        out_data = 1 / (1 + np.exp(-x.data))
        out = Tensor(out_data, (x,), 'Sigmoid')
        
        # Store output for backward pass
        self.cache = out_data
        
        def _backward():
            # Gradient: sigmoid(x) * (1 - sigmoid(x))
            grad = out.grad * (self.cache * (1 - self.cache))
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
        self.cache = None  # Store output for backward pass
    
    def forward(self, x):
        """
        Forward pass: tanh(x)
        
        Args:
            x: Input Tensor
        
        Returns:
            Tensor with tanh applied
        """
        # Compute tanh
        out_data = np.tanh(x.data)
        out = Tensor(out_data, (x,), 'Tanh')
        
        # Store output for backward pass
        self.cache = out_data
        
        def _backward():
            # Gradient: 1 - tanh^2(x)
            grad = out.grad * (1 - self.cache ** 2)
            x.grad += grad
        
        out._backward = _backward
        return out
    
    def __repr__(self):
        return "Tanh()"


class LeakyReLU(Module):
    """
    Leaky Rectified Linear Unit activation function
    f(x) = x if x > 0, alpha * x otherwise
    """
    
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha
        self.cache = None  # Store input for backward pass
    
    def forward(self, x):
        """
        Forward pass: LeakyReLU(x) = max(alpha*x, x)
        
        Args:
            x: Input Tensor
        
        Returns:
            Tensor with LeakyReLU applied
        """
        # Store input for backward pass
        self.cache = x
        
        # Apply LeakyReLU
        out_data = np.where(x.data > 0, x.data, self.alpha * x.data)
        out = Tensor(out_data, (x,), 'LeakyReLU')
        
        def _backward():
            # Gradient: 1 where x > 0, alpha elsewhere
            mask = np.where(self.cache.data > 0, 1.0, self.alpha)
            self.cache.grad += out.grad * mask
        
        out._backward = _backward
        return out
    
    def __repr__(self):
        return f"LeakyReLU(alpha={self.alpha})"


class ELU(Module):
    """
    Exponential Linear Unit activation function
    f(x) = x if x > 0, alpha * (exp(x) - 1) otherwise
    """
    
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.cache = None  # Store input and output for backward pass
    
    def forward(self, x):
        """
        Forward pass: ELU(x)
        
        Args:
            x: Input Tensor
        
        Returns:
            Tensor with ELU applied
        """
        # Store input for backward pass
        self.cache = x
        
        # Apply ELU
        out_data = np.where(
            x.data > 0, 
            x.data, 
            self.alpha * (np.exp(x.data) - 1)
        )
        out = Tensor(out_data, (x,), 'ELU')
        
        def _backward():
            # Gradient: 1 where x > 0, alpha * exp(x) elsewhere
            grad = np.where(
                self.cache.data > 0,
                1.0,
                self.alpha * np.exp(self.cache.data)
            )
            self.cache.grad += out.grad * grad
        
        out._backward = _backward
        return out
    
    def __repr__(self):
        return f"ELU(alpha={self.alpha})"


class Softmax(Module):
    """
    Softmax activation function for multi-class classification
    f(x_i) = exp(x_i) / sum(exp(x_j)) for all j
    """
    
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
        self.cache = None  # Store output for backward pass
    
    def forward(self, x):
        """
        Forward pass: softmax along specified dimension
        
        Args:
            x: Input Tensor
        
        Returns:
            Tensor with softmax applied
        """
        # Shift for numerical stability (subtract max)
        x_shifted = x.data - np.max(x.data, axis=self.dim, keepdims=True)
        
        # Compute exponentials
        exp_x = np.exp(x_shifted)
        
        # Compute softmax
        out_data = exp_x / np.sum(exp_x, axis=self.dim, keepdims=True)
        out = Tensor(out_data, (x,), 'Softmax')
        
        # Store output for backward pass
        self.cache = out_data
        
        def _backward():
            # Jacobian is more complex, but for cross-entropy loss,
            # the gradient simplifies. This is a general implementation.
            batch_size = out_data.shape[0]
            grad = out.grad.copy()
            
            # For each sample in batch
            for i in range(batch_size):
                s = out_data[i].reshape(-1, 1)
                # Jacobian: diag(s) - s @ s.T
                jacobian = np.diagflat(s) - np.dot(s, s.T)
                grad[i] = grad[i].reshape(1, -1) @ jacobian
            
            x.grad += grad.reshape(x.data.shape)
        
        out._backward = _backward
        return out
    
    def __repr__(self):
        return f"Softmax(dim={self.dim})"


class LogSoftmax(Module):
    """
    Log Softmax activation function
    log(softmax(x)) - more numerically stable
    """
    
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
        self.cache = None  # Store input for backward pass
    
    def forward(self, x):
        """
        Forward pass: log(softmax(x))
        
        Args:
            x: Input Tensor
        
        Returns:
            Tensor with log softmax applied
        """
        # Store input for backward pass
        self.cache = x
        
        # Shift for numerical stability
        x_shifted = x.data - np.max(x.data, axis=self.dim, keepdims=True)
        
        # Compute log softmax
        out_data = x_shifted - np.log(np.sum(np.exp(x_shifted), axis=self.dim, keepdims=True))
        out = Tensor(out_data, (x,), 'LogSoftmax')
        
        def _backward():
            # Gradient simplifies to: 1 - softmax(x)
            exp_x = np.exp(x.data - np.max(x.data, axis=self.dim, keepdims=True))
            softmax = exp_x / np.sum(exp_x, axis=self.dim, keepdims=True)
            
            # Gradient of log softmax
            grad = out.grad - np.sum(out.grad * softmax, axis=self.dim, keepdims=True)
            x.grad += grad
        
        out._backward = _backward
        return out
    
    def __repr__(self):
        return f"LogSoftmax(dim={self.dim})"


class Softplus(Module):
    """
    Softplus activation function: f(x) = log(1 + exp(x))
    Smooth approximation of ReLU
    """
    
    def __init__(self, beta=1.0, threshold=20.0):
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.cache = None  # Store input for backward pass
    
    def forward(self, x):
        """
        Forward pass: softplus(x)
        
        Args:
            x: Input Tensor
        
        Returns:
            Tensor with softplus applied
        """
        # Store input for backward pass
        self.cache = x
        
        # Apply softplus with numerical stability
        beta_x = self.beta * x.data
        out_data = np.where(
            beta_x > self.threshold,
            beta_x,
            np.log(1 + np.exp(beta_x))
        ) / self.beta
        
        out = Tensor(out_data, (x,), 'Softplus')
        
        def _backward():
            # Gradient: sigmoid(beta * x)
            beta_x = self.beta * self.cache.data
            sigmoid = 1 / (1 + np.exp(-beta_x))
            self.cache.grad += out.grad * sigmoid
        
        out._backward = _backward
        return out
    
    def __repr__(self):
        return f"Softplus(beta={self.beta})"


class Swish(Module):
    """
    Swish activation function: f(x) = x * sigmoid(x)
    """
    
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
        self.cache = None  # Store input and sigmoid for backward pass
    
    def forward(self, x):
        """
        Forward pass: swish(x) = x * sigmoid(beta * x)
        
        Args:
            x: Input Tensor
        
        Returns:
            Tensor with swish applied
        """
        # Compute sigmoid
        beta_x = self.beta * x.data
        sigmoid = 1 / (1 + np.exp(-beta_x))
        
        # Store for backward pass
        self.cache = (x.data, sigmoid)
        
        # Apply swish
        out_data = x.data * sigmoid
        out = Tensor(out_data, (x,), 'Swish')
        
        def _backward():
            x_data, sig = self.cache
            # Gradient: sigmoid + x * sigmoid * (1 - sigmoid)
            grad = out.grad * (sig + x_data * sig * (1 - sig))
            x.grad += grad
        
        out._backward = _backward
        return out
    
    def __repr__(self):
        return f"Swish(beta={self.beta})"


class GELU(Module):
    """
    Gaussian Error Linear Unit activation function
    f(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    
    def __init__(self):
        super().__init__()
        self.cache = None  # Store input for backward pass
    
    def forward(self, x):
        """
        Forward pass: GELU(x)
        
        Args:
            x: Input Tensor
        
        Returns:
            Tensor with GELU applied
        """
        # Store input for backward pass
        self.cache = x
        
        # GELU approximation
        sqrt_2_over_pi = np.sqrt(2 / np.pi)
        x_data = x.data
        x_cubed = x_data ** 3
        
        gelu_data = 0.5 * x_data * (1 + np.tanh(sqrt_2_over_pi * (x_data + 0.044715 * x_cubed)))
        out = Tensor(gelu_data, (x,), 'GELU')
        
        def _backward():
            # Approximate gradient (simplified)
            x_data = self.cache.data
            cdf = 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x_data + 0.044715 * x_data**3)))
            pdf = np.exp(-0.5 * x_data**2) / np.sqrt(2 * np.pi)
            self.cache.grad += out.grad * (cdf + x_data * pdf)
        
        out._backward = _backward
        return out
    
    def __repr__(self):
        return "GELU()"


class Identity(Module):
    """
    Identity activation function (no activation)
    f(x) = x
    """
    
    def forward(self, x):
        """
        Forward pass: identity(x)
        
        Args:
            x: Input Tensor
        
        Returns:
            Same as input
        """
        out = Tensor(x.data, (x,), 'Identity')
        
        def _backward():
            x.grad += out.grad
        
        out._backward = _backward
        return out
    
    def __repr__(self):
        return "Identity()"