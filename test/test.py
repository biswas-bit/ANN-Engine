from ann_engine.core.parameter import Parameter
from ann_engine.layers.base import Module
import numpy as np

class DummyLayer(Module):
    def __init__(self):
        super().__init__()
        self.w = Parameter(np.array([[1.,2.],[3.,4]]))
        self.b = Parameter(np.array([1., 2.]))
layer = DummyLayer()
params = layer.parameters()
print(params)