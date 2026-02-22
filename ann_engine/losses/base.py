from abc import ABC, abstractmethod

class Loss(ABC):
    def __init__(self, reduction='mean'):
        self.reduction = reduction
    
    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)
    
    @abstractmethod
    def forward(self, y_pred, y_true):
        pass
    
    def _reduct(self, loss):
        """ Apply reduction to the loss value"""
        if self.reduction == 'mean':
            return loss.mean(loss)
        elif self.reduction == 'sum':
            return loss.sum(loss)
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"invalid reduction type: {self.reduction}")
        
    