import numpy as np
from ann_engine.layers import ReLU, Sigmoid
from ann_engine.core import Tensor

sig = Sigmoid()
try:
  input_for_sig = Tensor(np.array([1.2,1.3, 1.4, 1.5]))
except Exception as e:
    print(f"error occoured : {e}")
    
try:
  print(sig.forward(input_for_sig))
  input_for_sig.backward()
  print(f" gradient for relu Activation: {input_for_sig.grad}")
  print(f" parent note for gradient : {input_for_sig._prev}")
  print(f" Operator for Relu in computation graph: {input_for_sig._op} ")
except Exception as e:
    print(f"error occoured: {e}")
    
input("press enter to exit...")
