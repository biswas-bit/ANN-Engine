import numpy as np
from ann_engine.losses.base import Loss
from ann_engine.core.tensor import Tensor


class MSELoss(Loss):
    """Mean Squared Error Loss"""
    
    def forward(self, y_pred, y_true):
        """
        Compute MSE loss: (y_pred - y_true)^2
        
        Args:
            y_pred: Tensor of predictions
            y_true: Tensor of targets (can be numpy array or Tensor)
        
        Returns:
            Tensor loss value
        """
        # Ensure y_true is a Tensor
        if not isinstance(y_true, Tensor):
            y_true = Tensor(y_true)
        
        # Compute squared error
        diff = y_pred - y_true
        squared_error = diff ** 2
        
        return self._reduce(squared_error)
    

class CrossEntropyLoss(Loss):
    """Cross Entropy Loss for classification"""
    
    def __init__(self, reduction='mean', epsilon=1e-12):
        super().__init__(reduction)
        self.epsilon = epsilon
    
    def forward(self, y_pred, y_true):
        """
        Compute cross entropy loss: -sum(y_true * log(y_pred))
        
        Args:
            y_pred: Tensor of predictions (probabilities, should sum to 1)
            y_true: Tensor of targets (one-hot encoded or class indices)
        
        Returns:
            Tensor loss value
        """
        
        if not isinstance(y_true, Tensor):
            y_true = Tensor(y_true)
        
        # Added small epsilon to avoid log(0)
        y_pred_clipped = y_pred.__class__(np.clip(y_pred.data, self.epsilon, 1 - self.epsilon))
        
        # Compute cross entropy
        if len(y_true.data.shape) == 1 or y_true.data.shape[-1] == 1:
            # Sparse labels (class indices)
            loss = -np.log(y_pred_clipped.data[range(len(y_pred.data)), y_true.data.astype(int)])
            loss = Tensor(loss)
        else:
            # One-hot encoded labels
            loss = -(y_true * y_pred_clipped.log())
        

        return self._reduce(loss)
    
class NLLLoss(Loss):
    """ Negative log likelihood loos """
    
    def forward(self, y_pred, y_true):
        """ Compute NLL loss: -y_true * log(y_pred)

        Args:
            y_pred : Tensor of Predictions (log probabilities)
            y_true : Tensor of targets 
            
        Returns:
            Tensor loss value
        """
        
        if not isinstance(y_true, Tensor):
            y_true = Tensor(y_true)
            
        # compute NLL Loss
        if len(y_true.data.shape) == 1 or y_true.data.shape[-1] == 1:
            loss = -y_pred.data[range(len(y_pred.data)), y_true.data.astype(int)]
            loss = Tensor(loss)
        
        else:
            loss = -(y_true * y_pred)
            
        # Apply reduction
        return self._reduce(loss)