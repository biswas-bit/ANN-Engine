from .activations import (
    ReLU, Sigmoid, Tanh, LeakyReLU, ELU,
    Softmax, LogSoftmax, Softplus, Swish, GELU, Identity
)

from .layers import Dense
from .sequential import Sequential

__all__ = [
    'ReLU',
    'Sigmoid', 
    'Tanh',
    'LeakyReLU',
    'ELU',
    'Softmax',
    'LogSoftmax',
    'Softplus',
    'Swish',
    'GELU',
    'Identity',
    'Dense',
    'Sequential',
]

__version__ = "0.1.0"
__doc__ = "Neural network layers module containing activations and dense layers"