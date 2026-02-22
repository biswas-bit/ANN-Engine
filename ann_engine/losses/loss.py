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
        
    
        diff = y_pred - y_true
        squared_error = diff ** 2
        
        # Apply reduction
        if self.reduction == 'mean':
            n_elements = np.prod(y_pred.data.shape)
            loss_value = squared_error.sum() / n_elements
            return loss_value
        elif self.reduction == 'sum':
            return squared_error.sum()
        elif self.reduction == 'none':
            return squared_error
        else:
            raise ValueError(f"Invalid reduction type: {self.reduction}")

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
    

class BCELoss(Loss):
    """ Binary Cross Entropy Loss"""
    def __init__(self, reduction = 'mean', epsilon=1e-12):
        super().__init__(reduction)
        self.epsilion  = epsilon
        
    def forward(self, y_pred, y_true):
        """
        Compute binary cross entropy loss:
        -[y_true * log(y_pred) + (1 - y_true)* log(1 - y_pred)]
        
        Args:
            y_pred: Tensor of Predictions (probabilities between 0 and 1)
            y_true: Tensor of targets (0 or 1)
             
        Returns:
            Tensor Loss value
        """
        if not isinstance(y_true, Tensor):
            y_true = Tensor(y_true)
            
            #clipping predictions to avoid log(0)
            y_pred_clipped = y_pred.__class__(np.clip(y_pred.data, self.epsilion, 1-self.epsilion))
            
            # computing binary cross entropy loss
            loss = -(y_true * y_pred_clipped.log() + (1 - y_true) * (1 - y_pred_clipped).log())
        
        return self._reduce(loss)
    

class BCEWithLogitsLoss(Loss):
    """ Binary Cross Entropy Loss with Logits"""
    def __init__(self, reduction='mean'):
        super().__init__(reduction)
        
    def forward(self, y_pred, y_true):
        """
        Compute BCE with logits loss:
        max(x, 0) - x * z + log(1 + exp(-abs(x)))
        
        Args:
            y_pred: Tensor of Predictions (logits)
            y_true: Tensor of targets (0 or 1)
            
        Returns:
            Tensor Loss value
        """
        if not isinstance(y_true, Tensor):
            y_true = Tensor(y_true)
        
        # compute BCE with logits loss
        loss = np.maximum(y_pred.data, 0) - y_pred.data * y_true.data + np.log(1 + np.exp(-np.abs(y_pred.data)))
        loss = Tensor(loss)
        
        return self._reduce(loss)
    

class HuberLoss(Loss):
    """ Huber Loss for regression """
    def __init__(self, reduction = 'mean', delta=1.0):
        super().__init__(reduction)
        self.delta = delta
        
    def forward(self, y_pred, y_true):
        """
        Compute Huber Loss
        0.5 * (y_pred - y_true)^2 if |y_pred - y_true| <= delta
        delta * (|y_pred - y_true| - 0.5 * delta^2) otherwise
        
        Args:
            y_pred: Tensor of Predictions
            y_true: Tensor of targets
        
        returns:
           Tensor Loss Value 
        """
        
        if not isinstance(y_true, Tensor):
            y_true = Tensor(y_true)
        
        diff = y_pred -y_true
        abs_diff = np.abs(diff)
        
        #quadratic loss for small errors, linear loss for large errors
        quadratic = np.minimum(abs_diff, self.delta)
        linear = abs_diff - quadratic
        loss = 0.5 * quadratic **2 + self.delta * linear
        
        return self._reduce(loss)
        