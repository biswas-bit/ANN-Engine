import numpy as np
from ann_engine.core.tensor import Tensor
from ann_engine.core import Parameter
from ann_engine.layers.base import Module


class Dense(Module):
    """Fully Connected layer
    y = xW + b

    Supports lazy initialization: if in_features is 0 or None,
    weights are created automatically on the first forward pass.
    """

    def __init__(self, in_features, out_features, bias=True, initialization='xavier'):
        super().__init__()

        self.in_features = in_features  # may be 0/None → lazy init
        self.out_features = out_features
        self.initialization = initialization
        self.use_bias = bias

        self.W = None
        self.b = None

        # Only initialize weights immediately if in_features is known
        if in_features:
            self._initialize_weights(in_features)

    # ------------------------------------------------------------------
    # Weight initialization
    # ------------------------------------------------------------------

    def _initialize_weights(self, in_features):
        """Create W and b given a concrete in_features value."""
        self.in_features = in_features

        if self.initialization == 'xavier':
            limit = np.sqrt(6 / (in_features + self.out_features))
            weight_data = np.random.uniform(-limit, limit, (in_features, self.out_features))
        elif self.initialization == 'he':
            std = np.sqrt(2 / in_features)
            weight_data = np.random.randn(in_features, self.out_features) * std
        elif self.initialization == 'normal':
            weight_data = np.random.randn(in_features, self.out_features) * 0.01
        elif self.initialization == 'uniform':
            limit = 1 / np.sqrt(in_features)
            weight_data = np.random.uniform(-limit, limit, (in_features, self.out_features))
        else:
            raise ValueError(f"Unknown initialization method: {self.initialization}")

        self.W = Parameter(weight_data)

        if self.use_bias:
            self.b = Parameter(np.zeros((1, self.out_features)))
        else:
            self.b = None

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch_size, in_features) or (in_features,)

        Returns:
            Tensor of shape (batch_size, out_features)
        """
        # Handle single-sample 1-D input
        if len(x.data.shape) == 1:
            x = x.reshape(1, -1)

        # ── Lazy initialization ──────────────────────────────────────
        if self.W is None:
            self._initialize_weights(x.data.shape[1])

        # ── Shape validation ─────────────────────────────────────────
        if x.data.shape[1] != self.in_features:
            raise ValueError(
                f"Expected input with {self.in_features} features, "
                f"but got {x.data.shape[1]} features"
            )

        # Matrix multiplication: (batch_size, in_features) @ (in_features, out_features)
        out = x @ self.W

        if self.b is not None:
            out = out + self.b

        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def parameters(self):
        """Return list of trainable parameters (only those initialized)."""
        params = []
        if self.W is not None:
            params.append(self.W)
        if self.b is not None:
            params.append(self.b)
        return params

    def __repr__(self):
        return (
            f"Dense(in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.use_bias}, "
            f"init={self.initialization})"
        )