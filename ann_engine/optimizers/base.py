class Optimizer:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr
    
    def step(self):
        raise NotImplementedError("This method should be implemented by subclasses")
    
    def zero_grad(self):
        for param in self.parameters:
            param.grad = 0 