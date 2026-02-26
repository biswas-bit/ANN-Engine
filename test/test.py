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
print("step 3: Forward pass - second call (x2)")
print("=" * 70)
out2 = softmax(x2)
print(f"Out2 data:{out2.data}")
print(f"Out2 Operation:{out2._op}")
print(f"Out2 parents:{[id(p) for p in out2._prev]}")
print(f"out2 id: {id(out2)}")

exp_x2 = np.exp(x2.data - np.max(x2.data))
expected2 = exp_x2 / np.sum(exp_x2)
print(f"Expected out2: {expected2}")
print(f"out2 correct ? {np.allclose(out2.data, expected2)}")

print("\n" + "=" * 70)
print("step 4: Forward pass - Third call (x3)")
print("=" * 70)
out3 = softmax(x3)
print(f"Out3 data:{out3.data}")
print(f"Out3 Operation:{out3._op}")
print(f"Out3 parents:{[id(p) for p in out3._prev]}")
print(f"out3 id: {id(out3)}")

exp_x3 = np.exp(x3.data - np.max(x3.data))
expected3 = exp_x3 / np.sum(exp_x3)
print(f"Expected out3: {expected3}")
print(f"out3 correct ? {np.allclose(out3.data, expected3)}")

print("\n" + "=" * 70)
print("step 5: Verify all Outputs are different")
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
print("STEP 9: Calculate Expected Gradients (CORRECTED)")
print("-" * 70)

# EXPLANATION:
# loss = sum(softmax(x))
# upstream grad into softmax = ones_like(softmax_output)  [from sum._backward]
# softmax grad: dx = s * (grad - sum(grad * s, keepdims=True))
#             = s * (ones - sum(ones * s))
#             = s * (ones - sum(s))          <- sum(s) along dim = 1.0 (softmax property)
#             = s * (ones - 1.0)
#             = s * 0 = 0
#
# So gradient of sum(softmax(x)) w.r.t x is ALWAYS ZERO.
# This is mathematically correct — softmax output always sums to 1,
# so the sum loss carries no information about x.
#
# To get non-zero gradients, use a meaningful loss like cross-entropy,
# or use a weighted sum (dot product with target).

def compute_expected_softmax_grad(x_data, upstream_grad):
    """Manually compute softmax backward given upstream gradient."""
    x_max = np.max(x_data, axis=-1, keepdims=True)
    exp_x = np.exp(x_data - x_max)
    s = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    sum_grad_s = np.sum(upstream_grad * s, axis=-1, keepdims=True)
    return s * (upstream_grad - sum_grad_s)

# From sum(), upstream grad is ones
upstream_ones = np.ones_like(out1.data)
expected_grad_x1 = compute_expected_softmax_grad(x1.data, upstream_ones)
expected_grad_x2 = compute_expected_softmax_grad(x2.data, upstream_ones)
expected_grad_x3 = compute_expected_softmax_grad(x3.data, upstream_ones)

print(f"Expected gradient for x1: {expected_grad_x1}")
print(f"Expected gradient for x2: {expected_grad_x2}")
print(f"Expected gradient for x3: {expected_grad_x3}")
print()
print("NOTE: All expected grads are ~0 because d/dx[sum(softmax(x))] = 0")
print("      softmax always sums to 1 regardless of x — the sum loss is constant!")
print("      Use cross-entropy or weighted loss for meaningful gradients.")

print("\n" + "-" * 70)
print("STEP 10: Compare Gradients")
print("-" * 70)

x1_correct = np.allclose(x1.grad, expected_grad_x1, atol=1e-6)
x2_correct = np.allclose(x2.grad, expected_grad_x2, atol=1e-6)
x3_correct = np.allclose(x3.grad, expected_grad_x3, atol=1e-6)

print(f"x1 gradient correct? {x1_correct}")
if not x1_correct:
    print(f"  Got:      {x1.grad}")
    print(f"  Expected: {expected_grad_x1}")

print(f"x2 gradient correct? {x2_correct}")
if not x2_correct:
    print(f"  Got:      {x2.grad}")
    print(f"  Expected: {expected_grad_x2}")

print(f"x3 gradient correct? {x3_correct}")
if not x3_correct:
    print(f"  Got:      {x3.grad}")
    print(f"  Expected: {expected_grad_x3}")

# For "grads different" test, use a MEANINGFUL loss instead
# Use weighted loss: loss = sum(out * weights) so grads are non-zero and different
print("\n" + "-" * 70)
print("STEP 10b: Verify with MEANINGFUL loss (weighted sum)")
print("-" * 70)

# Re-run with weighted loss so we get non-zero gradients
x1b = Parameter(np.array([[1.0, 2.0, 3.0]]))
x2b = Parameter(np.array([[4.0, 5.0, 6.0]]))
x3b = Parameter(np.array([[0.1, 0.2, 0.3]]))
softmax2 = Softmax()

out1b = softmax2(x1b)
out2b = softmax2(x2b)
out3b = softmax2(x3b)

# Use DIFFERENT weights per input so gradients are guaranteed to differ
weights1 = np.array([[0.1, 0.5, 0.9]], dtype=np.float32)
weights2 = np.array([[0.9, 0.2, 0.4]], dtype=np.float32)
weights3 = np.array([[0.3, 0.8, 0.1]], dtype=np.float32)

# loss = sum(out * weights)
loss_b = (out1b * Parameter(weights1)).sum() + \
         (out2b * Parameter(weights2)).sum() + \
         (out3b * Parameter(weights3)).sum()

loss_b.backward()

exp_grad_x1b = compute_expected_softmax_grad(x1b.data, weights1)
exp_grad_x2b = compute_expected_softmax_grad(x2b.data, weights2)
exp_grad_x3b = compute_expected_softmax_grad(x3b.data, weights3)

x1b_correct = np.allclose(x1b.grad, exp_grad_x1b, atol=1e-6)
x2b_correct = np.allclose(x2b.grad, exp_grad_x2b, atol=1e-6)
x3b_correct = np.allclose(x3b.grad, exp_grad_x3b, atol=1e-6)

print(f"x1b grad: {x1b.grad}")
print(f"x2b grad: {x2b.grad}")
print(f"x3b grad: {x3b.grad}")
print(f"x1b gradient correct? {x1b_correct}")
print(f"x2b gradient correct? {x2b_correct}")
print(f"x3b gradient correct? {x3b_correct}")

grads_different = (not np.allclose(x1b.grad, x2b.grad) and
                   not np.allclose(x1b.grad, x3b.grad) and
                   not np.allclose(x2b.grad, x3b.grad))
print(f"All gradients different? {grads_different}")

print("\n" + "-" * 70)
print("STEP 11: Isolated backward (only out1)")
print("-" * 70)

x1c = Parameter(np.array([[1.0, 2.0, 3.0]]))
x2c = Parameter(np.array([[4.0, 5.0, 6.0]]))
x3c = Parameter(np.array([[0.1, 0.2, 0.3]]))
softmax3 = Softmax()

out1c = softmax3(x1c)
out2c = softmax3(x2c)
out3c = softmax3(x3c)

# Only backprop through out1c
isolated_loss = (out1c * Parameter(weights1)).sum()
isolated_loss.backward()

print(f"x1c.grad: {x1c.grad} (should be non-zero)")
print(f"x2c.grad: {x2c.grad} (should be zero)")
print(f"x3c.grad: {x3c.grad} (should be zero)")

x1c_nonzero = not np.allclose(x1c.grad, 0)
x2c_zero = np.allclose(x2c.grad, 0)
x3c_zero = np.allclose(x3c.grad, 0)
print(f"x1c gradient non-zero? {x1c_nonzero}")
print(f"x2c gradient zero? {x2c_zero}")
print(f"x3c gradient zero? {x3c_zero}")

# ======================================================================
# DEPENDENT CALLS TESTS
# Chain: x -> softmax1 -> softmax2 -> softmax3 -> weighted loss
# Gradient must flow all the way back through all three softmax layers
# ======================================================================
print("\n" + "=" * 70)
print("STEP 12: Dependent calls - chained softmax")
print("=" * 70)
print("Chain: x_d -> softmax -> softmax -> softmax -> weighted loss")
print("Gradient must flow back through all 3 layers to x_d")

x_d = Parameter(np.array([[1.0, 2.0, 3.0]]))
softmax_d = Softmax()

# Three dependent calls — each feeds into the next
chain1 = softmax_d(x_d)          # softmax(x_d)
chain2 = softmax_d(chain1)       # softmax(softmax(x_d))
chain3 = softmax_d(chain2)       # softmax(softmax(softmax(x_d)))

weights_d = np.array([[0.1, 0.5, 0.9]], dtype=np.float32)
chain_loss = (chain3 * Parameter(weights_d)).sum()

print(f"\nx_d.grad before backward: {x_d.grad}")
print(f"chain1.data: {chain1.data}")
print(f"chain2.data: {chain2.data}")
print(f"chain3.data: {chain3.data}")

chain_loss.backward()

print(f"\nx_d.grad after backward:  {x_d.grad}")

# Manually compute expected gradient by chaining softmax jacobians
# dx = J3.T @ (J2.T @ (J1.T @ upstream))
# For softmax jacobian-vector product: J.T @ v = s * (v - sum(v*s))
def softmax_jvp(s, v):
    """Jacobian-vector product for softmax: s * (v - sum(v*s, keepdims))"""
    return s * (v - np.sum(v * s, axis=-1, keepdims=True))

s1 = chain1.data
s2 = chain2.data
s3 = chain3.data

# Backprop step by step
grad3 = weights_d                    # upstream into chain3 backward
grad2 = softmax_jvp(s3, grad3)      # grad flowing into chain2
grad1 = softmax_jvp(s2, grad2)      # grad flowing into chain1
grad_xd = softmax_jvp(s1, grad1)    # grad flowing into x_d

print(f"\nExpected x_d.grad:        {grad_xd}")
chain_correct = np.allclose(x_d.grad, grad_xd, atol=1e-6)
print(f"x_d gradient correct? {chain_correct}")

# ======================================================================
# STEP 13: Dependent + Independent mixed
# out_dep = softmax(softmax(x_e))
# out_ind = softmax(x_f)           <- independent
# loss = weighted(out_dep) + weighted(out_ind)
# x_e should get chained grad, x_f should get single-layer grad
# ======================================================================
print("\n" + "-" * 70)
print("STEP 13: Mixed dependent + independent calls")
print("-" * 70)
print("out_dep = softmax(softmax(x_e))  |  out_ind = softmax(x_f)")
print("Gradients must NOT leak between the two branches")

x_e = Parameter(np.array([[1.0, 2.0, 3.0]]))
x_f = Parameter(np.array([[0.5, 1.5, 2.5]]))
softmax_m = Softmax()

mid   = softmax_m(x_e)          # softmax(x_e)
out_dep = softmax_m(mid)        # softmax(softmax(x_e))
out_ind = softmax_m(x_f)        # softmax(x_f)  — independent

weights_e = np.array([[0.2, 0.6, 0.8]], dtype=np.float32)
weights_f = np.array([[0.7, 0.3, 0.5]], dtype=np.float32)

mixed_loss = (out_dep * Parameter(weights_e)).sum() + \
             (out_ind * Parameter(weights_f)).sum()

mixed_loss.backward()

# Expected for x_e: chain through two softmax layers
se1 = mid.data
se2 = out_dep.data
grad_dep_out = weights_e
grad_dep_mid = softmax_jvp(se2, grad_dep_out)
expected_grad_xe = softmax_jvp(se1, grad_dep_mid)

# Expected for x_f: single softmax layer
sf = out_ind.data
expected_grad_xf = softmax_jvp(sf, weights_f)

print(f"\nx_e.grad:          {x_e.grad}")
print(f"Expected x_e.grad: {expected_grad_xe}")
print(f"\nx_f.grad:          {x_f.grad}")
print(f"Expected x_f.grad: {expected_grad_xf}")

xe_correct = np.allclose(x_e.grad, expected_grad_xe, atol=1e-6)
xf_correct = np.allclose(x_f.grad, expected_grad_xf, atol=1e-6)
print(f"\nx_e gradient correct? {xe_correct}")
print(f"x_f gradient correct? {xf_correct}")

# Also verify no leakage: x_f grad should NOT equal x_e grad
no_leakage = not np.allclose(x_e.grad, x_f.grad)
print(f"No gradient leakage between branches? {no_leakage}")

# ======================================================================
# FINAL SUMMARY
# ======================================================================
print("\n" + "=" * 70)
all_passed = (x1_correct and x2_correct and x3_correct and
              x1b_correct and x2b_correct and x3b_correct and
              grads_different and x1c_nonzero and x2c_zero and x3c_zero and
              chain_correct and xe_correct and xf_correct and no_leakage)

if all_passed:
    print("✓✓✓ ALL TESTS PASSED! ✓✓✓")
    print("Softmax correctly handles independent, dependent, and mixed calls.")
else:
    print("✗✗✗ TESTS FAILED! ✗✗✗")
    if not (x1_correct and x2_correct and x3_correct):
        print("  - Basic gradients incorrect (sum loss)")
    if not (x1b_correct and x2b_correct and x3b_correct):
        print("  - Weighted loss gradients incorrect")
    if not grads_different:
        print("  - Gradients not all different")
    if not x1c_nonzero:
        print("  - Isolated x1 grad is zero (should be non-zero)")
    if not x2c_zero or not x3c_zero:
        print("  - Gradient leaking between independent calls")
    if not chain_correct:
        print("  - Chained (dependent) gradient incorrect")
    if not xe_correct:
        print("  - Mixed dependent branch gradient incorrect")
    if not xf_correct:
        print("  - Mixed independent branch gradient incorrect")
    if not no_leakage:
        print("  - Gradient leaking between mixed branches")
print("=" * 70)

input("press enter ..")