from .tensor import Tensor

class Parameter(Tensor):
    """A Parameter is a Tensor that should be optimized"""
    def __init__(self, data):
        super().__init__(data)