from .tensor import Tensor

class Parameter(Tensor):
    """ 
    Trainable tensor
    used for weights and biases inside layers
    """
    def __init__(self, data, required_grad=True):
        super().__init__(data)
        self.required_grad = required_grad