from abc import ABC, abstractmethod

class Optimizer:
    def __init__(self, lr=0.01):
        self.parameters = None
        self.lr = lr
        
    def set_parameters(self, parameters):
        """ set the Parameters to optimize called by model.compile """
        self.parameters = parameters
        
    @abstractmethod
    def step(self):
        raise NotImplementedError("This method should be implemented by subclasses")
    
    @abstractmethod
    def zero_grad(self):
        for param in self.parameters:
            param.grad = 0 