import numpy as np
from ann_engine.losses.base import Loss
from ann_engine.core.tensor import Tensor


class MSELoss(Loss):
    """Mean Squared Error Loss"""

    def forward(self, y_pred, y_true):
        if not isinstance(y_true, Tensor):
            y_true = Tensor(y_true)

        squared_error = (y_pred - y_true) ** 2

        if self.reduction == 'mean':
            n_elements = np.prod(y_pred.data.shape)
            return squared_error.sum() / n_elements
        elif self.reduction == 'sum':
            return squared_error.sum()
        elif self.reduction == 'none':
            return squared_error
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")


class CrossEntropyLoss(Loss):
    """
    Categorical Cross-Entropy Loss.

    Uses Tensor ops throughout so gradients flow back through y_pred.
    Raw numpy ops (np.log, np.sum) would orphan the loss tensor from
    the computation graph and produce zero gradients everywhere.
    """

    def __init__(self, reduction='mean', epsilon=1e-8):
        super().__init__(reduction)
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        if not isinstance(y_true, Tensor):
            y_true = Tensor(y_true)

        # Convert sparse integer labels → one-hot
        if y_true.data.ndim == 1 or (
            y_true.data.ndim == 2 and y_true.data.shape[1] == 1
        ):
            batch_size  = y_pred.data.shape[0]
            num_classes = y_pred.data.shape[1]
            indices     = y_true.data.astype(int).flatten()
            one_hot     = np.zeros((batch_size, num_classes), dtype=np.float32)
            one_hot[np.arange(batch_size), indices] = 1.0
            y_true = Tensor(one_hot)

        # ── KEY: use Tensor.clip() + Tensor.log() ────────────────────
        # These methods attach _backward hooks and keep the node
        # connected to y_pred in the computation graph.
        # Never use np.clip(y_pred.data) / np.log(y_pred.data) here —
        # that breaks the graph and gives zero gradients.
        clipped     = y_pred.clip(self.epsilon, 1.0 - self.epsilon)
        log_pred    = clipped.log()
        per_element = y_true * log_pred          # (batch, classes)
        total       = per_element.sum()          # scalar
        loss        = total * (-1.0 / float(y_pred.data.shape[0]))

        return loss


class BCELoss(Loss):
    """
    Binary Cross-Entropy Loss.

    Uses Tensor.clip() + Tensor.log() to keep the computation graph
    intact. The old y_pred.__class__(np.clip(...)) pattern created an
    orphan tensor with no _prev, breaking backprop entirely.
    """

    def __init__(self, reduction='mean', epsilon=1e-8):
        super().__init__(reduction)
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        if not isinstance(y_true, Tensor):
            y_true = Tensor(y_true)

        # ── KEY: Tensor.clip() keeps graph connected ──────────────────
        # y_pred.__class__(np.clip(...)) was the bug — it produced a
        # brand-new Tensor with no link back to y_pred, so backward()
        # never reached the model weights.
        clipped = y_pred.clip(self.epsilon, 1.0 - self.epsilon)

        # -(y * log(p) + (1-y) * log(1-p))
        one        = Tensor(np.ones_like(y_true.data))
        loss       = -(y_true * clipped.log() + (one - y_true) * (one - clipped).log())

        return self._reduce(loss)


class BCEWithLogitsLoss(Loss):
    """
    Binary Cross-Entropy with Logits (numerically stable).

    NOTE: this loss uses raw numpy internally because the numerically
    stable formula doesn't have a clean Tensor decomposition. It is
    intentionally disconnected from the graph — use only when your
    final layer outputs raw logits (no sigmoid), and you don't need
    gradients flowing through the loss itself (e.g. inference only),
    OR replace with BCELoss + Sigmoid output for full backprop.
    """

    def forward(self, y_pred, y_true):
        if not isinstance(y_true, Tensor):
            y_true = Tensor(y_true)

        # Numerically stable: max(x,0) - x*z + log(1 + exp(-|x|))
        x = y_pred.data
        z = y_true.data
        loss_data = np.maximum(x, 0) - x * z + np.log(1 + np.exp(-np.abs(x)))
        return self._reduce(Tensor(loss_data))


class NLLLoss(Loss):
    """Negative Log-Likelihood Loss (expects log-probabilities as input)."""

    def forward(self, y_pred, y_true):
        if not isinstance(y_true, Tensor):
            y_true = Tensor(y_true)

        if y_true.data.ndim == 1 or y_true.data.shape[-1] == 1:
            batch_size = len(y_pred.data)
            loss_data  = -y_pred.data[range(batch_size), y_true.data.astype(int).flatten()]
            return self._reduce(Tensor(loss_data))
        else:
            return self._reduce(-(y_true * y_pred))


class HuberLoss(Loss):
    """Huber Loss — quadratic for small errors, linear for large ones."""

    def __init__(self, reduction='mean', delta=1.0):
        super().__init__(reduction)
        self.delta = delta

    def forward(self, y_pred, y_true):
        if not isinstance(y_true, Tensor):
            y_true = Tensor(y_true)

        diff     = y_pred - y_true
        abs_diff = np.abs(diff.data)
        mask     = abs_diff <= self.delta

        loss_data = np.where(
            mask,
            0.5 * diff.data ** 2,
            self.delta * abs_diff - 0.5 * self.delta ** 2
        )
        return self._reduce(Tensor(loss_data))