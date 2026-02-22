class Module:
    """ Base Class for all neural network modules """
    
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        
    def forward(self,*inputs):
        raise NotImplementedError("Forward method not implemented")
    
    def __call__(self, *inputs):
        return self.forward(*inputs)
    
    def __setattr__(self, name, value):
        from ann_engine.core.parameter import Parameter
        from ann_engine.layers.base import Module
        
        if isinstance(value, Parameter):
            self._parameters[name] = value
        
        elif isinstance(value, Module):
            self._modules[name] = value
            
        super().__setattr__(name, value)
        
    def parameters(self):
        params = list(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        
        return params