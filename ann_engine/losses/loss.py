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
        return self._reduct(squared_error)
    

    