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
        if not isinstance(loss, Tensor):
            loss = Tensor(loss)
        
        if self.reduction == 'mean':
            # Use loss.data for shape information
            n_elements = np.prod(loss.data.shape)
            if n_elements == 1:
                return loss
            else:
                sum_loss = loss.sum()
                return Tensor(sum_loss.data / n_elements)
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Invalid reduction type: {self.reduction}")