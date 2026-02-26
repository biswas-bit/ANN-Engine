from ann_engine.layers import Dense as BaseDense
from ann_engine.layers import ReLU, Sigmoid, Tanh, Softmax

class Dense(BaseDense):
    """
    Dense Layer with built-in activation
    
    Example:
         Dense(128, activation='relu')
       
    """
    def __init__(self, units, activation=None, use_bias=True, input_shape=None, **kwargs):
        self.input_shape = input_shape
        self.units = units
        self.activation_name = activation
        
        in_features = input_shape[-1] if input_shape else None
        super().__init__(in_features or 0, units, bias = use_bias, **kwargs)
        
        # set activation
        self.activation = self._getactivation(activation)
        
    
    def _get_activation(self, name):
        """ Get activation function """
        if name is None or name == 'linear':
            return None
        elif name =='relu':
            return ReLU()
        elif name == 'sigmoid':
            return Sigmoid
        elif name == 'tanh':
            return Tanh()
        elif name == 'softmax':
            return Softmax()
        else:
            raise ValueError(f"Unknown activation : {name}")
        
    def forward(self,x):
        x = super().forward(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
    
    def __repr__(self):
        return (f"Dense(units={self.units}, "
                f"activation={self.activation_name})")
        
        