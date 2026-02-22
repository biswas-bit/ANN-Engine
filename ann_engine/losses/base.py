from abc import ABC, abstractmethod
import numpy as np
from ann_engine.core.tensor import Tensor

class Loss(ABC):
    def __init__(self, reduction='mean'):
        self.reduction = reduction
    
    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)
    
    @abstractmethod
    def forward(self, y_pred, y_true):
        pass
    
    def _reduce(self, loss):
        """Apply reduction to the loss value"""
        if self.reduction == 'mean':
            # Use loss.data.shape to get the shape as a tuple of integers
            n_elements = np.prod(loss.data.shape)
            return loss.sum() / n_elements
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Invalid reduction type: {self.reduction}")