import numpy as np
import sys
import traceback

def pause_on_error():
    print("\n" + "=" * 60)
    input("Press Enter to continue after error...")
    

try:
    from ann_engine.core import Tensor, Parameter
    from ann_engine.layers import ELU  # Changed to ELU
except ImportError as e:
    print(f"Failed to Import libraries : {e}")
    pause_on_error()
    sys.exit(1)
    
def run_test(test_name, test_func):
    print(f"\n" + "=" * 50)
    print(f"Running test : {test_name}")
    print(f"\n" + "=" * 50)
    
    try:
        result = test_func()
        print(f"\n ✓ Test Passed : {test_name}")
        return True, result
    except AssertionError as e:
        print(f"\n ✗ Test Failed (Assertion Error): {test_name}")
        print(f" {e}")
        print("\n" + "-" * 50)
        traceback.print_exc()
        pause_on_error()
        return False, None
    except Exception as e:
        print(f'\n ✗ Test Failed (Exception): {test_name}')
        print(f" {e}")
        print("\n" + "-" * 50)
        traceback.print_exc()
        pause_on_error()
        return False, None
    

def test_elu_forward_basic():
    """ Test 1: Basic ELU forward pass with default alpha=1.0 """
    print("Testing basic ELU forward pass")
    
    elu = ELU()  # Default alpha = 1.0
    print(f"ELU instance: {elu}")
    
    # Test with various values
    x_data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    x = Tensor(x_data)
    out = elu(x)
    
    # Expected result: x if x > 0, alpha * (exp(x) - 1) otherwise
    alpha = 1.0
    expected = np.where(x_data > 0, x_data, alpha * (np.exp(x_data) - 1))
    
    print(f"Input: {x_data}")
    print(f"Output: {out.data}")
    print(f"Expected (alpha={alpha}): {expected}")
    
    assert np.allclose(out.data, expected), f"Expected {expected}, got {out.data}"
    assert out._op == "ELU", f"Expected op 'ELU', got {out._op}"
    assert x in out._prev, "Input should be in computation graph"
    print("✓ Forward pass correct")
    print("✓ Computation graph built correctly")
    return True

def test_elu_forward_custom_alpha():
    """ Test 2: ELU forward pass with custom alpha """
    print("Testing ELU forward pass with custom alpha=0.5")
    
    alpha = 0.5
    elu = ELU(alpha=alpha)
    print(f"ELU instance with alpha={alpha}: {elu}")
    
    x_data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    x = Tensor(x_data)
    out = elu(x)
    
    # Expected result: x if x > 0, alpha * (exp(x) - 1) otherwise
    expected = np.where(x_data > 0, x_data, alpha * (np.exp(x_data) - 1))
    
    print(f"Input: {x_data}")
    print(f"Output: {out.data}")
    print(f"Expected (alpha={alpha}): {expected}")
    
    assert np.allclose(out.data, expected), f"Expected {expected}, got {out.data}"
    print("✓ Custom alpha forward pass correct")
    return True

def test_elu_forward_2d():
    """ Test 3: ELU forward pass with 2D input """
    print("Testing ELU forward pass with 2D input...")
    
    alpha = 1.0
    elu = ELU(alpha=alpha)
    
    x_data = np.array([
        [-1.0, -0.5, 0.0, 0.5],
        [1.0, 1.5, -2.0, -1.0],
        [0.0, -3.0, 4.0, 5.0]
    ])
    
    x = Tensor(x_data)
    out = elu(x)
    expected = np.where(x_data > 0, x_data, alpha * (np.exp(x_data) - 1))
    
    print(f"Input shape: {x_data.shape}")
    print(f"Output shape: {out.data.shape}")
    print(f"Sample output:\n{out.data}")
    
    assert out.data.shape == x_data.shape, f"Shape mismatch: {out.data.shape} vs {x_data.shape}"
    assert np.allclose(out.data, expected), "Output doesn't match expected"
    print("✓ 2D forward pass correct")
    return True

def test_elu_backward_basic():
    """ Test 4: Basic ELU backward pass """
    print("Testing ELU backward pass with basic values...")
    
    alpha = 1.0
    elu = ELU(alpha=alpha)
    
    x = Tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    out = elu(x)
    
    out.grad = np.ones_like(out.data)
    out._backward()
    
    # Expected gradient: 1 where x > 0, alpha * exp(x) elsewhere
    expected_grad = np.where(x.data > 0, 1.0, alpha * np.exp(x.data))
    
    print(f"Input: {x.data}")
    print(f"Output: {out.data}")
    print(f"Computed gradients: {x.grad}")
    print(f"Expected gradients (alpha={alpha}): {expected_grad}")
    
    assert np.allclose(x.grad, expected_grad), f"Expected {expected_grad}, got {x.grad}"
    print("✓ Backward pass correct")
    
    return True

def test_elu_backward_custom_alpha():
    """ Test 5: ELU backward pass with custom alpha """
    print("Testing ELU backward pass with custom alpha=0.5...")
    
    alpha = 0.5
    elu = ELU(alpha=alpha)
    
    x = Tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    out = elu(x)
    
    out.grad = np.ones_like(out.data)
    out._backward()
    
    # Expected gradient: 1 where x > 0, alpha * exp(x) elsewhere
    expected_grad = np.where(x.data > 0, 1.0, alpha * np.exp(x.data))
    
    print(f"Input: {x.data}")
    print(f"Output: {out.data}")
    print(f"Computed gradients: {x.grad}")
    print(f"Expected gradients (alpha={alpha}): {expected_grad}")
    
    assert np.allclose(x.grad, expected_grad), f"Expected {expected_grad}, got {x.grad}"
    print("✓ Custom alpha backward pass correct")
    
    return True

def test_elu_backward_with_loss():
    """ Test 6: ELU backward pass with loss.backward() """
    print("Testing ELU backward pass with loss.backward()...")
    
    alpha = 1.0
    elu = ELU(alpha=alpha)
    
    x = Parameter(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    out = elu(x)
    loss = out.sum()
    loss.backward()
    
    # Expected gradient: 1 where x > 0, alpha * exp(x) elsewhere
    expected_grad = np.where(x.data > 0, 1.0, alpha * np.exp(x.data))
    
    print(f"Input: {x.data}")
    print(f"Output: {out.data}")
    print(f"Loss: {loss.data}")
    print(f"Computed gradients: {x.grad}")
    print(f"Expected gradients: {expected_grad}")
    
    assert np.allclose(x.grad, expected_grad), f"Expected {expected_grad}, got {x.grad}"
    print("✓ Backward pass with loss.backward() correct")
    return True

def test_elu_backward_2d():
    """ Test 7: ELU backward pass with 2D input """
    print("Testing ELU backward pass with 2D input...")
    
    alpha = 1.0
    elu = ELU(alpha=alpha)
    
    x_data = np.array([
        [-1.0, -0.5, 0.0, 0.5],
        [1.0, 1.5, -2.0, -1.0],
        [0.0, -3.0, 4.0, 5.0]
    ])
    x = Parameter(x_data)
    out = elu(x)
    loss = out.sum()
    loss.backward()
    
    # Expected gradient: 1 where x > 0, alpha * exp(x) elsewhere
    expected_grad = np.where(x_data > 0, 1.0, alpha * np.exp(x_data))
    
    print(f"Input shape: {x_data.shape}")
    print(f"Output shape: {out.data.shape}")
    print(f"Computed gradients shape: {x.grad.shape}")
    print(f"Sample gradients:\n{x.grad}")
    
    assert x.grad.shape == x_data.shape, f"Gradient shape mismatch: {x.grad.shape} vs {x_data.shape}"
    assert np.allclose(x.grad, expected_grad), "Gradients don't match expected"
    print("✓ 2D backward pass correct")
    
    return True

def test_elu_computation_graph():
    """ Test 8: Verify computation graph is built correctly """
    print("Testing ELU computation graph...")
    
    elu = ELU()
    
    x = Tensor(np.array([-1.0, 0.0, 1.0]))
    y = Tensor(np.array([2.0, 2.0, 2.0]))
    
    out = elu(x) + y
    print(f"Input x: {x.data}")
    print(f"Input y: {y.data}")
    print(f"Output: {out.data}")
    print(f"Output operation: {out._op}")
    print(f"Output parents: {[id(p) for p in out._prev]}")
    
    assert out._op == '+', "Output should be from addition"
    
    # Find the ELU node in the graph
    elu_node = None
    for child in out._prev:
        if child._op == 'ELU':
            elu_node = child
            break
    
    assert elu_node is not None, "ELU node not found in computation graph"
    assert x in elu_node._prev, "Input x should be parent of ELU node"
    print("✓ Computation graph built correctly")
    
    return True

def test_elu_edge_cases():
    """ Test 9: ELU edge cases (large values) """
    print("Testing ELU edge cases...")
    
    alpha = 1.0
    elu = ELU(alpha=alpha)
    
    # Test with various values
    x_large = Parameter(np.array([100.0, -100.0, 10.0, -10.0]))
    out_large = elu(x_large)
    
    print(f"Large values input: {x_large.data}")
    print(f"Output: {out_large.data}")
    
    # For large positive x, output ≈ x
    # For large negative x, output ≈ -alpha (since exp(-100) ≈ 0, so alpha*(0-1) = -alpha)
    # But note: exp(-10) is not exactly 0, it's a very small number
    expected_output = np.array([
        100.0,                    # 100 > 0, so output = 100
        -alpha,                    # -100: alpha*(exp(-100)-1) ≈ alpha*(0-1) = -alpha
        10.0,                      # 10 > 0, so output = 10
        alpha * (np.exp(-10.0) - 1) # -10: exact calculation
    ])
    
    print(f"Expected output: {expected_output}")
    
    # Check large positive
    assert np.allclose(out_large.data[0], expected_output[0]), f"Large positive: got {out_large.data[0]}, expected {expected_output[0]}"
    
    # Check large negative (should be close to -alpha)
    assert np.allclose(out_large.data[1], -alpha, rtol=1e-3), f"Large negative: got {out_large.data[1]}, expected ~{-alpha}"
    
    # Check medium positive
    assert np.allclose(out_large.data[2], 10.0), f"Medium positive: got {out_large.data[2]}, expected 10.0"
    
    # Check medium negative (exact calculation)
    assert np.allclose(out_large.data[3], alpha * (np.exp(-10.0) - 1)), f"Medium negative incorrect"
    
    print("✓ Forward pass edge cases correct")
    
    # Now test backward pass
    loss_large = out_large.sum()
    loss_large.backward()
    
    print(f"Gradients: {x_large.grad}")
    
    # Expected gradients:
    # For large positive: 1.0
    # For large negative: alpha * exp(-100) ≈ 0
    # For medium positive: 1.0
    # For medium negative: alpha * exp(-10)
    expected_grad = np.array([
        1.0,                          # x=100: gradient = 1
        alpha * np.exp(-100.0),        # x=-100: gradient ≈ 0
        1.0,                          # x=10: gradient = 1
        alpha * np.exp(-10.0)          # x=-10: exact gradient
    ])
    
    print(f"Expected gradients: {expected_grad}")
    
    # Check gradients with appropriate tolerances
    assert np.allclose(x_large.grad[0], 1.0), f"Gradient for large positive: got {x_large.grad[0]}, expected 1.0"
    assert np.allclose(x_large.grad[1], 0.0, atol=1e-10), f"Gradient for large negative: got {x_large.grad[1]}, expected ~0"
    assert np.allclose(x_large.grad[2], 1.0), f"Gradient for medium positive: got {x_large.grad[2]}, expected 1.0"
    assert np.allclose(x_large.grad[3], alpha * np.exp(-10.0)), f"Gradient for medium negative incorrect"
    
    print("✓ Backward pass edge cases correct")
    print("✓ Edge cases handled correctly")
    
    return True

def test_elu_properties():
    """ Test 10: ELU special properties """
    print("Testing ELU properties...")
    
    alpha = 2.0
    elu = ELU(alpha=alpha)
    
    # Test 1: For negative values, ELU approaches -alpha as x -> -inf
    x_neg_large = Parameter(np.array([-100.0]))
    out_neg_large = elu(x_neg_large)
    assert np.allclose(out_neg_large.data, -alpha), f"ELU(-inf) should approach -{alpha}"
    print(f"✓ ELU(-inf) approaches -{alpha}")
    
    # Test 2: ELU(0) = 0
    x_zero = Parameter(np.array([0.0]))
    out_zero = elu(x_zero)
    assert np.allclose(out_zero.data, 0.0), "ELU(0) should be 0"
    print("✓ ELU(0) = 0")
    
    # Test 3: For positive values, ELU(x) = x
    x_pos = Parameter(np.array([5.0]))
    out_pos = elu(x_pos)
    assert np.allclose(out_pos.data, 5.0), "ELU(x) should equal x for x>0"
    print("✓ ELU(x) = x for x>0")
    
    # Test 4: Gradient for negative values is alpha * exp(x)
    x_neg = Parameter(np.array([-1.0]))
    out_neg = elu(x_neg)
    loss_neg = out_neg.sum()
    loss_neg.backward()
    expected_grad_neg = alpha * np.exp(-1.0)
    assert np.allclose(x_neg.grad, expected_grad_neg), f"Gradient for negative should be {expected_grad_neg}"
    print(f"✓ Gradient for negative values = alpha * exp(x)")
    
    return True

def test_elu_different_alphas():
    """ Test 11: Different alpha values """
    print("Testing ELU with different alpha values...")
    
    alphas = [0.5, 1.0, 2.0, 3.0]
    x_data = np.array([-2.0, -1.0, 1.0, 2.0])
    
    for alpha in alphas:
        elu = ELU(alpha=alpha)
        x = Tensor(x_data)
        out = elu(x)
        
        expected = np.where(x_data > 0, x_data, alpha * (np.exp(x_data) - 1))
        
        print(f"Alpha={alpha}: {out.data}")
        assert np.allclose(out.data, expected), f"Failed for alpha={alpha}"
    
    print("✓ All alpha values work correctly")
    return True

def test_elu_numerical_stability():
    """ Test 12: ELU numerical stability """
    print("Testing ELU numerical stability...")
    
    alpha = 1.0
    elu = ELU(alpha=alpha)
    
    # Test with very small negative values
    x_small_neg = Parameter(np.array([-1e-10, -1e-20]))
    out_small_neg = elu(x_small_neg)
    
    print(f"Very small negative input: {x_small_neg.data}")
    print(f"Output: {out_small_neg.data}")
    
    # Should not produce NaN or Inf
    assert not np.any(np.isnan(out_small_neg.data)), "Output contains NaN"
    assert not np.any(np.isinf(out_small_neg.data)), "Output contains Inf"
    print("✓ Numerical stability for small negative values")
    
    # Test with very large negative values
    x_large_neg = Parameter(np.array([-100.0, -200.0]))
    out_large_neg = elu(x_large_neg)
    
    print(f"Very large negative input: {x_large_neg.data}")
    print(f"Output: {out_large_neg.data}")
    
    # Should saturate at -alpha
    expected = np.array([-alpha, -alpha])
    assert np.allclose(out_large_neg.data, expected), f"Should saturate at -{alpha}"
    print(f"✓ Saturates at -{alpha} for large negative values")
    
    return True

def test_elu_multiple_uses():
    """ Test 13: Same ELU instance used multiple times """
    print("Testing ELU instance used multiple times...")
    
    alpha = 1.0
    elu = ELU(alpha=alpha)
    
    # Create different inputs
    x1 = Parameter(np.array([-1.0, 0.0, 1.0]))
    x2 = Parameter(np.array([-2.0, 2.0, -3.0]))
    
    print(f"\nx1 input: {x1.data}")
    print(f"x2 input: {x2.data}")
    
    # Use same ELU instance twice
    out1 = elu(x1)
    out2 = elu(x2)
    
    print(f"\nout1 (ELU of x1): {out1.data}")
    print(f"out2 (ELU of x2): {out2.data}")
    
    # Calculate expected values manually
    expected_out1 = np.where(x1.data > 0, x1.data, alpha * (np.exp(x1.data) - 1))
    expected_out2 = np.where(x2.data > 0, x2.data, alpha * (np.exp(x2.data) - 1))
    
    print(f"\nExpected out1: {expected_out1}")
    print(f"Expected out2: {expected_out2}")
    
    # Verify forward passes are correct and different
    assert np.allclose(out1.data, expected_out1), "out1 forward pass incorrect"
    assert np.allclose(out2.data, expected_out2), "out2 forward pass incorrect"
    assert not np.allclose(out1.data, out2.data), "Outputs should be different for different inputs"
    print("✓ Forward passes correct and different")
    
    # Create separate losses
    loss1 = out1.sum()
    loss2 = out2.sum()
    loss = loss1 + loss2
    
    print(f"\nloss1: {loss1.data}")
    print(f"loss2: {loss2.data}")
    print(f"total loss: {loss.data}")
    
    # Backward pass
    loss.backward()
    
    print(f"\nx1 gradients: {x1.grad}")
    print(f"x2 gradients: {x2.grad}")
    
    # Expected gradients: 1 where x > 0, alpha * exp(x) elsewhere
    expected_grad_x1 = np.where(x1.data > 0, 1.0, alpha * np.exp(x1.data))
    expected_grad_x2 = np.where(x2.data > 0, 1.0, alpha * np.exp(x2.data))
    
    print(f"\nExpected x1 gradients: {expected_grad_x1}")
    print(f"Expected x2 gradients: {expected_grad_x2}")
    
    # Check if gradients are correct
    x1_correct = np.allclose(x1.grad, expected_grad_x1)
    x2_correct = np.allclose(x2.grad, expected_grad_x2)
    gradients_different = not np.allclose(x1.grad, x2.grad)
    
    print(f"\nx1 correct? {x1_correct}")
    print(f"x2 correct? {x2_correct}")
    print(f"gradients different? {gradients_different}")
    
    if not x1_correct:
        print(f"x1 grad diff: {x1.grad - expected_grad_x1}")
    if not x2_correct:
        print(f"x2 grad diff: {x2.grad - expected_grad_x2}")
    
    assert gradients_different, "Gradients should be different for different inputs"
    assert x1_correct, f"x1 gradients: got {x1.grad}, expected {expected_grad_x1}"
    assert x2_correct, f"x2 gradients: got {x2.grad}, expected {expected_grad_x2}"
    
    print("\n✓ Multiple uses of same ELU instance works correctly")
    return True

def main():
    tests = [
    ('Forward Pass Basic', test_elu_forward_basic),
    ('Forward Pass Custom Alpha', test_elu_forward_custom_alpha),
    ('Forward Pass 2D', test_elu_forward_2d),
    ('Backward Pass Basic', test_elu_backward_basic),
    ('Backward Pass Custom Alpha', test_elu_backward_custom_alpha),
    ('Backward with Loss', test_elu_backward_with_loss),
    ('Backward Pass 2D', test_elu_backward_2d),
    ('Computation Graph', test_elu_computation_graph),
    ('Edge Cases', test_elu_edge_cases),  
    ('ELU Properties', test_elu_properties),
    ('Different Alphas', test_elu_different_alphas),
    ('Numerical Stability', test_elu_numerical_stability),
    ('Multiple Uses', test_elu_multiple_uses),
]
    passed = 0
    failed = 0
    failed_tests = []
    
    for test_name, test_func in tests:
        success, _ = run_test(test_name, test_func)
        if success:
            passed += 1
        else:
            failed += 1
            failed_tests.append(test_name)
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY - ELU")
    print("=" * 50)
    print(f"Total Tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed_tests:
        print("\nFailed tests:")
        for test in failed_tests:
            print(f"  ✗ {test}")
    
    print("\n" + "=" * 50)
    
    if failed == 0:
        print("\n🎉 ALL ELU TESTS PASSED 🎉")
        print("ELU implementation is working correctly!")
    else:
        print(f"\n❌ {failed} Test(s) Failed. Check the errors above.")
        print("The program paused after each error so you can see what happened.")
    
    print("\n" + "=" * 50)
    input("\nPress Enter to exit...") 

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
        input("\nPress Enter to exit...")
    except Exception as e:
        print(f"\nUnexpected error in main: {e}")
        traceback.print_exc()
        input("\nPress Enter to exit")