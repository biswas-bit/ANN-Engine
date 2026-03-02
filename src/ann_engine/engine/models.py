# ann_engine/nn/model.py
from ann_engine.layers import Sequential

class Model(Sequential):
    """
    Model class for custom architectures (similar to tf.keras.Model)
    
    Example:
        class MyModel(Model):
            def __init__(self):
                super().__init__()
                self.fc1 = Dense(128, activation='relu')
                self.fc2 = Dense(64, activation='relu')
                self.fc3 = Dense(10, activation='softmax')
            
            def forward(self, x):
                x = self.fc1(x)
                x = self.fc2(x)
                x = self.fc3(x)
                return x
    """
    
    def __init__(self):
        super().__init__([])
        self._layers = []
    
    def add_layer(self, layer):
        """Add a layer to the model"""
        self._layers.append(layer)
    
    def forward(self, x):
        """Override this method"""
        raise NotImplementedError("Subclasses must implement forward()")
    
    def __call__(self, x):
        return self.forward(x)
    
    def parameters(self):
        """Get all parameters"""
        params = []
        for layer in self._layers:
            params.extend(layer.parameters())
        return params