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


class LeakyReLU(Module):
    """
    Leaky Rectified Linear Unit activation function
    f(x) = x if x>0 , alpha * x otherwise
    
    """
    def __init__(self):
        super().__init__()
        self.cache = None
        
    def forward(self, x):
        """
        forward pass :
        LeakyReLU(x) = max(alpha*x, x) 

        Args:
            x : Input Tensor
            
        Returns:
           Tensor with LeakyReLU applied
           
        """
        
        self.cache = x
        out_data = np.where(x.data > 0 , x.data, self.alpha * x.data)
        out = Tensor(out_data, (x,), 'LeakyReLU')
        
        def _backward():
            # Grdient: 1 where x > 0 , alpha otherwise
            mask = np.where(self.cache.data > 0, 1.0, self.alpha)
            self.cache.grad += out.grad * mask
        out._backward = _backward
        return out
    
    def __repr__(self):
        return f"LeakyReLU (alpha = {self.alpha})"
    

class ELU(Module):
    """
    Exponential Linear Unit activation function
    f(x) = x if x> 0, alpha*(axp(x)-1) otherwise
    """
    
    def __init__(self):
        super().__init__()
        self.cache = None
        
    def forward(self,x):
        """
        forward pass : ELU(x)
        
        Args:
           x : Input Tensor
           
        Returns:
         Tensor with ELU
         
        """
        self.cache = x
        out_data = np.where(
            x.data > 0,
            x.data,
            self.alpha * (np.exp(x.data)-1)
        )
        
        out = Tensor(out_data, (x,), 'ELU')
        
        def _backward():
              # Gradient: 1 where x > 0, alpha * exp(x) elsewhere
              grad = np.where(
                  self.cache.data > 0,
                  1.0,
                  self.alpha * np.exp(self.cache.data))
              self.cache.grad += out.grad * grad
              
        out._backward = _backward
        return out
    
    def __repr__(self):
        return f"ELU(alpha={self.alpha})"
    

class Softmax(Module):
    """
    softmax activation function for multi-class classification
    f(x_i) = exp(X_i) / sum(exp(x_j)) for all j
    
    """
    
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
        self.cache = None
        
    def forward(self,x):
        """Forward pass : softmax along specfied dimension

        Args:
            x : Input Tensor
            
        Returns:
           Tensor with softmax applied
           
        """
        # shift for numerical stability (subtract max)
        x_shifted = x.data - np.max(x.data, axis=self.dim, keepdims=True)
        exp_x = np.exp(x_shifted)
        
        # compute softmax
        out_data  = exp_x / np.sum(exp_x, axis=self.dim, keepdims=True)
        out = Tensor(out_data, (x,), 'Softmax')
        
        self.cache = out_data
        
        def _backward():
            # jacobian is more complex, but for cross-entropy loss,
            # the gradient simplifies . This is a general implementation.
            
            batch_size = out_data.shape[0]
            grad = out.grad.copy()
            
            for i in range(batch_size):
                s = out[i].reshape(-1,1)
                jacobian = np.diagflat(s) - np.dot(s, s.T)
                grad[i] = grad[i].reshape(1,-1) @ jacobian
            x.grad += grad.reshape(x.data.shape)
            out._backward = _backward
            return out
        
        def __repr__(self):
            return f"Softmax(dim={self.dim})"
        
class Logsoftmax(Module):
    """
      Log Softmax activation function
      log(softmax(x)) - more numerically stable
    """
    
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
        self.cache = None 
        
    def forward(self,x):
        """
        forward Pass : log(softmax(x))

        Args:
            x : Input Tensor
        
        Returns:
            Tensor with log softmax applied 
        """
        self.cache = x
        x_shifted = x.data - np.max(x.data, axis=self.dim, keepdims=True)
        
        out_data = x_shifted - np.log(np.sum(np.exp(x_shifted), axis=self.dim, keepdim=True))
        out = Tensor(out_data, (x,), 'LogSoftmax')
        
        def _backward():
            # Gradient simplifies to : 1- softmax(x)
            exp_x = np.exp(x.data - np.max(x.data, axis=self.dim, keepdims=True))
            softmax = exp_x / np.sum(exp_x, axis=self.dim, keepdim=True)
            
            # Gradient of log softmax
            grad = out.grad - np.sum(out.grad * softmax, axis=self.dim, keepdims=True)
            x.grad += grad
        out._backward = _backward
        return out
    
class softplus(Module):
    """
      softplus activation function : f(x) = log(1+ exp(x))
      smooth approximation of ReLU
      
    """
    def __init__(self, bets=1.0 , threshold=20.0):
        super().__init__()
        self.beta = self.beta
        self.threshold = threshold
        self.cache = None
        
    def forward(self, x):
        """
        forward pass : softplus(x)
        
        Args:
           x: Input Tensor
           
        Returns:
           Tensor with softplus applied
            
        """
        
        self.cache = x
        beta_x = self.beta * x.data 
        out_data = np.where (
            beta_x > self.threshold,
            beta_x,
            np.log(1 + np.exp(beta_x))
        ) / self.beta
        
        out = Tensor(out_data, (x,), 'softplus')
        
        def _backward():
            beta_x = self.beta * self.cache.data
            sigmoid = 1/(1 + np.exp(-beta_x))
            self.cache.grad += out.grad * sigmoid
        out._backward = _backward
        return out
        
    def __repr__(self):
        return f" Softplus (beta={self.beta})"
            
        
