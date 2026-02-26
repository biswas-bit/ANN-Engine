

from ann_engine.layers import Softmax
from ann_engine.core import Parameter
import numpy as np

x1 = Parameter(np.array([[1.0, 2.0, 3.0]]))
x2 = Parameter(np.array([[4.0, 5.0, 6.0]])) 
x3 = Parameter(np.array([[0.1, 0.2, 0.3]]))

print("\n" + "=" * 70)
print("step 1: Input value")
print("=" * 70)
print(f"x1 data: {x1.data} | shape {x1.data.shape} | id: {id(x1)}")
print(f"x2 data: {x2.data} | shape {x2.data.shape} | id: {id(x2)}")
print(f"x3 data: {x3.data} | shape {x3.data.shape} | id: {id(x3)}")

print("\n" + "=" * 70)
print("step 2: Forward pass - First call (x1)")
print("=" * 70)

softmax = Softmax()
out1 = softmax(x1)
print(f"Out1 data:{out1.data}")
print(f"Out1 Operation:{out1._op}")
print(f"Out1 parents:{[id(p) for p in out1._prev]}")
print(f"out1 id: {id(out1)}")

exp_x1 = np.exp(x1.data - np.max(x1.data))
expected1 = exp_x1 / np.sum(exp_x1)
print(f"Expected out1: {expected1}")
print(f"out1 correct ? {np.allclose(out1.data, expected1)}")

input("press enter ..")