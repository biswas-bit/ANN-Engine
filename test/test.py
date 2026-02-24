import numpy as np
from ann_engine.layers import ReLU, Sigmoid, Tanh
from ann_engine.core import Tensor

tan = Tanh()
try:
  input_for_tan = Tensor(np.array([1.2,1.3, 1.4, 1.5]))
except Exception as e:
    print(f"error occoured : {e}")
    
try:
  print(tan.forward(input_for_tan))
  input_for_tan.backward()
  print(f" gradient for relu Activation: {input_for_tan.grad}")
  print(f" parent note for gradient : {input_for_tan._prev}")
  print(f" Operator for Relu in computation graph: {input_for_tan._op} ")
except Exception as e:
    print(f"error occoured: {e}")
    
input("press enter to exit...")
