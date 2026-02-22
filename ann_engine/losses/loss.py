import numpy as np
from ann_engine.losses.base import Loss
from ann_engine.core.tensor import Tensor

class MSE(Loss):
    """ Mean Squared Error Loss"""
    def forward(self, y_pred, y_true):
        if not isinstance(y_pred, Tensor):
            y_pred = Tensor(y_true)
        
        diff = y_pred - y_true
        squared_error = diff**2
        return self._reduce(squared_error)
    

class CrossEntropy(Loss):
    """ cross entropy loss for multi-class classification"""
    def __init__(self, reduction='mean', epsilon=1e-12):
        super().__init__(reduction)
        self.epsilion = epsilon
        
    def forward(self, y_pred, y_true):
        if not isinstance(y_pred, Tensor):
            y_pred = Tensor(y_pred)
            
        y_pred_clipped = y_pred.__class__(np.clip(y_pred.data, self.epsilon, 1-self.epsilon))
        if len(y_true.data.shape) ==1 or y_true.data.shape[-1] ==1:
            loss = -np.log(y_pred_clipped.data[range(len(y_true.data)), y_true.data.astype(int)])
            loss = Tensor(loss)
        else:
            loss -=(y_true*y_pred_clipped.log())
        return self._reduct(loss)