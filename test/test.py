

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

print("\n" + "=" * 70)
print("step 3: Forward pass - second call (x1)")
print("=" * 70)

out2 = softmax(x2)
print(f"Out2 data:{out2.data}")
print(f"Out2 Operation:{out2._op}")
print(f"Out2 parents:{[id(p) for p in out2._prev]}")
print(f"out2 id: {id(out2)}")

exp_x2 = np.exp(x2.data - np.max(x2.data))
expected2 = exp_x2 / np.sum(exp_x2)
print(f"Expected out2: {expected2}")
print(f"out1 correct ? {np.allclose(out1.data, expected2)}")



print("\n" + "=" * 70)
print("step 4: Forward pass - Third call (x1)")
print("=" * 70)

out3 = softmax(x3)
print(f"Out3 data:{out3.data}")
print(f"Out3 Operation:{out3._op}")
print(f"Out3 parents:{[id(p) for p in out3._prev]}")
print(f"out3 id: {id(out3)}")

exp_x3 = np.exp(x3.data - np.max(x3.data))
expected3 = exp_x3 / np.sum(exp_x3)
print(f"Expected out3: {expected3}")
print(f"out1 correct ? {np.allclose(out3.data, expected3)}")


print("\n" + "=" * 70)
print("step 5: Forward pass - Veryfy all Outputs are differentiable")
print("=" * 70)
out1_vs_out2_diff = not np.allclose(out1.data, out2.data)
out1_vs_out3_diff = not np.allclose(out1.data, out3.data)
out2_vs_out3_diff = not np.allclose(out2.data, out3.data)

print(f"out1 vs out2 different? {out1_vs_out2_diff}")
print(f"out1 vs out3 different? {out1_vs_out3_diff}")
print(f"out2 vs out3 different? {out2_vs_out3_diff}")


input("press enter ..")