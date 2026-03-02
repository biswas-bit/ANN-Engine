from ann_engine.layers.dense import Dense as BaseDense
from ann_engine.layers import ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, ELU


class Dense(BaseDense):
    """
    Dense layer with built-in activation and optional input_shape.

    input_shape is optional — when omitted, in_features is inferred
    lazily from the first input tensor (Keras-style).

    Example:
        Dense(128, activation='relu', input_shape=(784,))  # first layer
        Dense(64,  activation='relu')                      # subsequent layers
        Dense(10,  activation='softmax')
    """

    def __init__(self, units, activation=None, use_bias=True,
                 input_shape=None, **kwargs):

        self.units           = units
        self.activation_name = activation

        # Resolve in_features from input_shape — 0 means lazy init
        if input_shape is not None:
            if isinstance(input_shape, (tuple, list)):
                in_features = input_shape[-1]
            else:
                in_features = int(input_shape)
        else:
            in_features = 0

        super().__init__(in_features, units, bias=use_bias, **kwargs)

        self.activation = self._get_activation(activation)

    # ------------------------------------------------------------------

    def _get_activation(self, name):
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
            raise ValueError(
                f"Unknown activation: '{name}'. "
                f"Choose from {list(activations.keys())}"
            )
        return activations[name]()

    # ------------------------------------------------------------------

    def forward(self, x):
        x = super().forward(x)           # BaseDense: lazy init + matmul + bias
        if self.activation is not None:
            x = self.activation(x)
        return x

    # ------------------------------------------------------------------
    # parameters() is inherited from BaseDense — no override needed

    def __repr__(self):
        return (
            f"Dense(units={self.units}, "
            f"in_features={self.in_features}, "
            f"activation='{self.activation_name}')"
        )