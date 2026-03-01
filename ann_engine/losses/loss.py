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
        squared_error = (y_pred - y_true) ** 2
        
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
    """
    Categorical Cross-Entropy Loss.

    Supports:
        - One-hot encoded labels  : y_true shape (batch, num_classes)
        - Sparse integer labels   : y_true shape (batch,) or (batch, 1)

    IMPORTANT: uses Tensor ops throughout so gradients flow back
    through y_pred correctly. Raw numpy ops would orphan the loss
    tensor from the computation graph and produce zero gradients.
    """

    def __init__(self, reduction='mean', epsilon=1e-8):
        super().__init__(reduction)
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred : Tensor, shape (batch_size, num_classes)
                     — softmax probabilities output by the model
            y_true : Tensor or array, one-hot (batch, classes)
                     or sparse integer labels (batch,)

        Returns:
            Scalar Tensor loss value
        """
        if not isinstance(y_true, Tensor):
            y_true = Tensor(y_true)

        # ── Convert sparse labels → one-hot ──────────────────────────
        if y_true.data.ndim == 1 or (
            y_true.data.ndim == 2 and y_true.data.shape[1] == 1
        ):
            batch_size  = y_pred.data.shape[0]
            num_classes = y_pred.data.shape[1]
            indices     = y_true.data.astype(int).flatten()
            one_hot     = np.zeros((batch_size, num_classes), dtype=np.float32)
            one_hot[np.arange(batch_size), indices] = 1.0
            y_true = Tensor(one_hot)   # plain Tensor, no grad needed for labels

       
        clipped = y_pred.clip(self.epsilon, 1.0 - self.epsilon)

       
        log_pred = clipped.log()

        
        per_element = y_true * log_pred         
        total       = per_element.sum()          
        batch_size  = float(y_pred.data.shape[0])

        # Negate and average over batch
        loss = total * (-1.0 / batch_size)

        return loss


class NLLLoss(Loss):
    """Negative Log Likelihood Loss"""
    
    def forward(self, y_pred, y_true):
        """
        Compute NLL loss: -log(y_pred[y_true])
        
        Args:
            y_pred: Tensor of predictions (log probabilities)
            y_true: Tensor of targets (class indices)
            
        Returns:
            Tensor loss value
        """
        if not isinstance(y_true, Tensor):
            y_true = Tensor(y_true)
        
        # Compute NLL Loss
        if len(y_true.data.shape) == 1 or y_true.data.shape[-1] == 1:
            # Sparse labels (class indices)
            batch_size = len(y_pred.data)
            loss_data = -y_pred.data[range(batch_size), y_true.data.astype(int)]
            loss = Tensor(loss_data)
        else:
            # One-hot encoded labels
            loss = -(y_true * y_pred)
        
        # Apply reduction
        return self._reduce(loss)


class BCELoss(Loss):
    """Binary Cross Entropy Loss"""
    
    def __init__(self, reduction='mean', epsilon=1e-12):
        super().__init__(reduction)
        self.epsilon = epsilon  # FIXED: was 'epsilion'
    
    def forward(self, y_pred, y_true):
        """
        Compute binary cross entropy loss:
        -[y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)]
        
        Args:
            y_pred: Tensor of predictions (probabilities between 0 and 1)
            y_true: Tensor of targets (0 or 1)
             
        Returns:
            Tensor loss value
        """
        if not isinstance(y_true, Tensor):
            y_true = Tensor(y_true)
        
        # FIXED: Unindented these lines
        # Clipping predictions to avoid log(0)
        y_pred_clipped = y_pred.__class__(np.clip(y_pred.data, self.epsilon, 1 - self.epsilon))
        
        # Computing binary cross entropy loss
        loss = -(y_true * y_pred_clipped.log() + (1 - y_true) * (1 - y_pred_clipped).log())
        
        return self._reduce(loss)


class BCEWithLogitsLoss(Loss):
    """Binary Cross Entropy Loss with Logits (numerically stable)"""
    
    def __init__(self, reduction='mean'):
        super().__init__(reduction)
    
    def forward(self, y_pred, y_true):
        """
        Compute BCE with logits loss using numerically stable formula:
        max(x, 0) - x * z + log(1 + exp(-abs(x)))
        
        where x = logits, z = targets
        
        Args:
            y_pred: Tensor of predictions (logits, not probabilities)
            y_true: Tensor of targets (0 or 1)
            
        Returns:
            Tensor loss value
        """
        if not isinstance(y_true, Tensor):
            y_true = Tensor(y_true)
        
        # Numerically stable BCE with logits
        loss_data = (np.maximum(y_pred.data, 0) - 
                     y_pred.data * y_true.data + 
                     np.log(1 + np.exp(-np.abs(y_pred.data))))
        loss = Tensor(loss_data)
        
        return self._reduce(loss)


class HuberLoss(Loss):
    """Huber Loss for robust regression"""
    
    def __init__(self, reduction='mean', delta=1.0):
        super().__init__(reduction)
        self.delta = delta
    
    def forward(self, y_pred, y_true):
        """
        Compute Huber Loss:
        - If |diff| <= delta: loss = 0.5 * diff^2
        - If |diff| > delta:  loss = delta * |diff| - 0.5 * delta^2
        
        Args:
            y_pred: Tensor of predictions
            y_true: Tensor of targets
        
        Returns:
            Tensor loss value
        """
        if not isinstance(y_true, Tensor):
            y_true = Tensor(y_true)
        
        # FIXED: Correct implementation
        diff = y_pred - y_true
        abs_diff = np.abs(diff.data)  # FIXED: use diff.data
        
        # Create mask for quadratic vs linear region
        mask = abs_diff <= self.delta
        
        # Quadratic region: 0.5 * diff^2
        quadratic = 0.5 * diff.data ** 2
        
        # Linear region: delta * |diff| - 0.5 * delta^2
        linear = self.delta * abs_diff - 0.5 * self.delta ** 2
        
        # Apply mask to select correct formula
        loss_data = np.where(mask, quadratic, linear)
        loss = Tensor(loss_data)
        
        return self._reduce(loss)