from ann_engine.core import Tensor
from ann_engine.layers.base import Module as LayerModule
import numpy as np

class Sequential:
    """
    Sequential Model that stacks layer sequentially similar to tf.keras.sequential
    
    Example:
     model = Sequential([
            Dense(128, activation='relu', input_shape=(784,)),
            Dense(64, activation='relu'),
            Dense(10, activation='softmax')
        ])
    """  
    
    def __init__(self, layers=None):
        self.layers = []
        self.built = False
        self._loss = None
        self._optimizer = None
        self._metrics = []
        
        if layers is not None:
            for layer in layers:
                self.add(layer)
                
    def add(self, layer):
        """ add layer to the model """
        self.layers.append(layer)
        self.built = False
        
    def build(self, input_shape):
        """ BUild the model by initilizing all layers """
        if self.built:
            return
        
        x = Tensor(np.zeros((1, *input_shape[1:])))
        for layer in self.layers:
            x = layer(x)
            
        self.built = True
        self._input_shape = input_shape
        self._output_shape = x.data.shape 
        
    def forward(self,x):
        """ forward pass through all layers"""
        if not self.built:
            self.build(x.data.shape)
            
        for layer in self.layers:
            x = layer(x)
            
        return x
    
    def __call__(self,x):
        return self.forward(x)
               