

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

print("\n" + "-" * 70)
print("STEP 6: Create Combined Loss")
print("-" * 70)
loss1 = out1.sum()
loss2 = out2.sum()
loss3 = out3.sum()
total_loss = loss1 + loss2 + loss3

print(f"loss1 (sum of out1): {loss1.data}")
print(f"loss2 (sum of out2): {loss2.data}")
print(f"loss3 (sum of out3): {loss3.data}")
print(f"total_loss: {total_loss.data}")

print("\n" + "-" * 70)
print("STEP 7: Check Gradients Before Backward")
print("-" * 70)
print(f"x1.grad before: {x1.grad}")
print(f"x2.grad before: {x2.grad}")
print(f"x3.grad before: {x3.grad}")


print("\n" + "-" * 70)
print("STEP 8: Backward Pass")
print("-" * 70)
total_loss.backward()
    
print(f"x1.grad after: {x1.grad}")
print(f"x2.grad after: {x2.grad}")
print(f"x3.grad after: {x3.grad}")

print("\n" + "-" * 70)
print("STEP 9: Calculate Expected Gradients")
print("-" * 70)

grad_upstream = 1.0
s1 = out1.data
sum_s1 = np.sum(s1, axis=-1, keepdims=True)
expected_grad_x1 = s1 * (grad_upstream - s1)

s2 = out2.data
expected_grad_x2 = s2 * (grad_upstream - s2)

s3 = out3.data
expected_grad_x3 = s3 * (grad_upstream - s3)

print(f"Expected gradient for x1: {expected_grad_x1 }")
print(f"Expected gradient for x2: {expected_grad_x2}")
print(f"Expected gradient for x3: {expected_grad_x3}")


print("\n" + "-" * 70)
print("STEP 10: Compare Gradients")
print("-" * 70)

x1_correct = np.allclose(x1.grad, expected_grad_x1, rtol=1e-5)
x2_correct = np.allclose(x2.grad, expected_grad_x2, rtol=1e-5)
x3_correct = np.allclose(x3.grad, expected_grad_x3, rtol=1e-5)

print(f"x1 gradient correct? {x1_correct}")
if not x1_correct:
    print(f"  Difference: {x1.grad - expected_grad_x1}")
    print(f"  Got: {x1.grad}")
    print(f"  Expected: {expected_grad_x1}")
    
print(f"x2 gradient correct? {x2_correct}")
if not x2_correct:
    print(f"  Difference: {x2.grad - expected_grad_x2}")
    print(f"  Got: {x2.grad}")
    print(f"  Expected: {expected_grad_x2}")
    
print(f"x3 gradient correct? {x3_correct}")
if not x3_correct:
    print(f"  Difference: {x3.grad - expected_grad_x3}")
    print(f"  Got: {x3.grad}")
    print(f"  Expected: {expected_grad_x3}")
    
grads_different = (not np.allclose(x1.grad, x2.grad) and 
                       not np.allclose(x1.grad, x3.grad) and 
                       not np.allclose(x2.grad, x3.grad))
    
print(f"\nAll gradients different? {grads_different}")
    
    


input("press enter ..")