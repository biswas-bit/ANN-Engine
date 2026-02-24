import numpy as np
from ann_engine.layers import ReLU
from ann_engine.core import Tensor

relu = ReLU()
try:
  input_for_relu = Tensor(np.array([1.2,1.3, 1.4, 1.5]))
except Exception as e:
    print(f"error occoured : {e}")
try:
  print(relu.forward(input_for_relu))
  input_for_relu.backward()
  print(f" gradient for relu Activation: {input_for_relu.grad}")
  print(f" parent note for gradient : {input_for_relu._prev}")
  print(f" Operator for Relu in computation graph: {input_for_relu._op} ")
except Exception as e:
    print(f"error occoured: {e}")
    
input("press enter to exit...")
