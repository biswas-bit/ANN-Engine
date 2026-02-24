import numpy as np
from ann_engine.layers import ReLU, Sigmoid, Tanh, LeakyReLU
from ann_engine.core import Tensor

Lr = LeakyReLU(alpha=0.10)
try:
  input_for_lr = Tensor(np.array([1.2,1.3, 1.4, 1.5]))
except Exception as e:
    print(f"error occoured : {e}")
    
try:
  print(Lr.forward(input_for_lr))
  input_for_lr.backward()
  print(f" gradient for relu Activation: {input_for_lr.grad}")
  print(f" parent note for gradient : {input_for_lr._prev}")
  print(f" Operator for Relu in computation graph: {input_for_lr._op} ")
except Exception as e:
    print(f"error occoured: {e}")
    
input("press enter to exit...")
