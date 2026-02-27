from ann_engine.layers import Sequential

class Model(Sequential):
    """ model class for custon architecture """
    
    def __init__(self):
        super().__init__([])
        self._layers = []
        
    def add_layer(self, layer):
        """Add a layer to the model"""
        self._layers.append(layer)
        
    def forward(self,x):
        raise NotImplementedError ("subclass Must implement forward()")
    
    
    def parameters(self):
        """ Get all Parameters"""
        params = []
        for layer in self._layers:
            params.extend(layer.parameters())
        return params