import numpy as np
from ann_engine.core.tensor import Tensor
from ann_engine.core import Parameter
from ann_engine.layers.base import Module

class Dense(Module):
    """Fully Connected layer
    y = xW + b
    """
    
    def __init__(self, in_features, out_features, bias=True, initialization='xavier'):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.initialization = initialization
        
        # Initialize weights based on method
        if initialization == 'xavier':
            limit = np.sqrt(6 / (in_features + out_features))
            weight_data = np.random.uniform(-limit, limit, (in_features, out_features))
        elif initialization == 'he':
            std = np.sqrt(2 / in_features)
            weight_data = np.random.randn(in_features, out_features) * std
        elif initialization == 'normal':
            weight_data = np.random.randn(in_features, out_features) * 0.01
        elif initialization == 'uniform':
            limit = 1 / np.sqrt(in_features)
            weight_data = np.random.uniform(-limit, limit, (in_features, out_features))
        else:
            raise ValueError(f"Unknown initialization method: {initialization}")
        
        self.W = Parameter(weight_data)
        
        if bias:
            # Initialize bias to zeros (better for gradients)
            bias_data = np.zeros((1, out_features))
            self.b = Parameter(bias_data)
        else:
            self.b = None
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Tensor of shape (batch_size, in_features) or (in_features,) for single sample
        
        Returns:
            Tensor of shape (batch_size, out_features)
        """
        # Store original shape for potential debugging
        original_shape = x.data.shape
        
        # Handle single sample case (1D input)
        if len(x.data.shape) == 1:
            x = x.reshape(1, -1)
        
        # Check input features
        if x.data.shape[1] != self.in_features:
            raise ValueError(
                f"Expected input with {self.in_features} features, "
                f"but got {x.data.shape[1]} features"
            )
        
        # Matrix multiplication: (batch_size, in_features) @ (in_features, out_features)
        out = x @ self.W
        
        # Add bias if present - handle broadcasting explicitly
        if self.b is not None:
            # Ensure bias is broadcastable
            batch_size = out.data.shape[0]
            # Reshape bias to (1, out_features) if needed
            if len(self.b.data.shape) == 1:
                self.b.data = self.b.data.reshape(1, -1)
            out = out + self.b
        
        return out
    
    def __repr__(self):
        return (f"Dense(in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.b is not None}, "
                f"init={self.initialization})")