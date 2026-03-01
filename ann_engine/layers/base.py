# ann_engine/layers/base.py
from abc import ABC, abstractmethod
from ann_engine.core import Parameter

class Module(ABC):
    def __init__(self):
        self._parameters = []
        self.training = True
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
    
    def parameters(self):
        """Return list of all parameters in the module"""
        params = []
        for attr in dir(self):
            obj = getattr(self, attr)
            if isinstance(obj, Parameter):
                params.append(obj)
            elif isinstance(obj, Module):
                params.extend(obj.parameters())
        return params
    
    def train(self):
        """Set module to training mode"""
        self.training = True
        for child in self.children():
            child.train()
    
    def eval(self):
        """Set module to evaluation mode"""
        self.training = False
        for child in self.children():
            child.eval()
    
    def children(self):
        """Return child modules"""
        children = []
        for attr in dir(self):
            obj = getattr(self, attr)
            if isinstance(obj, Module):
                children.append(obj)
        return children
    
    def parameters(self):
        params = list(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        return params