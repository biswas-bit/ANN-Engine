import numpy as np
from ann_engine.core.tensor import Tensor
from ann_engine.core import Parameter
from ann_engine.layers.base import Module


class Dense(Module):
    """
    Fully Connected layer:  out = x @ W + b

    Supports lazy initialization — if in_features is 0 or None,
    weights are created automatically on the first forward pass.
    """

    def __init__(self, in_features, out_features, bias=True, initialization='xavier'):
        super().__init__()

        self.in_features    = in_features
        self.out_features   = out_features
        self.initialization = initialization
        self.use_bias       = bias

        self.W = None
        self.b = None

        if in_features:
            self._initialize_weights(in_features)

    # ------------------------------------------------------------------
    # Weight initialization
    # ------------------------------------------------------------------

    def _initialize_weights(self, in_features):
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
            raise ValueError(f"Unknown initialization: '{self.initialization}'")

        self.W = Parameter(weight_data)
        self.b = Parameter(np.zeros((1, self.out_features))) if self.use_bias else None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)

        if x.data.ndim == 1:
            x = x.reshape(1, -1)

        # Lazy init on first real input
        if self.W is None:
            self._initialize_weights(x.data.shape[1])

        if x.data.shape[1] != self.in_features:
            raise ValueError(
                f"Expected {self.in_features} input features, "
                f"got {x.data.shape[1]}"
            )

        out = x @ self.W
        if self.b is not None:
            out = out + self.b
        return out

    # ------------------------------------------------------------------
    # Parameters — explicit override so optimizer always finds W and b
    # ------------------------------------------------------------------

    def parameters(self):
        """Return [W, b] — only those that have been initialized."""
        params = []
        if self.W is not None:
            params.append(self.W)
        if self.b is not None:
            params.append(self.b)
        return params

    def __repr__(self):
        return (
            f"Dense(in={self.in_features}, out={self.out_features}, "
            f"bias={self.use_bias}, init={self.initialization})"
        )