from ann_engine.layers.dense import Dense as BaseDense
from ann_engine.layers import ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, ELU


class Dense(BaseDense):
    """
    Dense layer with built-in activation and optional input_shape.

    When input_shape is omitted (the common case in a Sequential stack),
    in_features is left as 0 and the parent Dense will lazily initialize
    weights on the very first forward pass.

    Example:
        # Explicit input shape on first layer only:
        Dense(128, activation='relu', input_shape=(784,))

        # All subsequent layers — no input_shape needed:
        Dense(64, activation='relu')
        Dense(10, activation='softmax')
    """

    def __init__(self, units, activation=None, use_bias=True,
                 input_shape=None, **kwargs):

        self.units = units
        self.activation_name = activation

        # Resolve in_features from input_shape (may remain 0 → lazy init)
        if input_shape is not None:
            if isinstance(input_shape, (tuple, list)):
                # (784,) → 784  |  (None, 784) → 784
                in_features = input_shape[-1]
            else:
                in_features = int(input_shape)
        else:
            in_features = 0          # triggers lazy init in BaseDense

        super().__init__(in_features, units, bias=use_bias, **kwargs)

        self.activation = self._get_activation(activation)

    # ------------------------------------------------------------------

    def _get_activation(self, name):
        """Map activation name → activation module."""
        if name is None or name == 'linear':
            return None
        activations = {
            'relu':       ReLU,
            'sigmoid':    Sigmoid,
            'tanh':       Tanh,
            'softmax':    Softmax,
            'leaky_relu': lambda: LeakyReLU(alpha=0.01),
            'elu':        lambda: ELU(alpha=1.0),
        }
        if name not in activations:
            raise ValueError(f"Unknown activation: '{name}'. "
                             f"Choose from {list(activations.keys())}")
        return activations[name]()

    # ------------------------------------------------------------------

    def forward(self, x):
        # BaseDense.forward handles lazy init + matmul + bias
        x = super().forward(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"Dense(units={self.units}, "
            f"in_features={self.in_features}, "
            f"activation='{self.activation_name}')"
        )