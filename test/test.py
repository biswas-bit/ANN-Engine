import numpy as np
import sys
import traceback

def pause_on_error():
    print("\n" + "=" * 60)
    input("Press Enter to continue after error...")
    

try:
    from ann_engine.core import Tensor, Parameter
    from ann_engine.layers import LeakyReLU  # Changed to LeakyReLU
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
    

def test_leakyrelu_forward_basic():
    """ Test 1: Basic LeakyReLU forward pass with default alpha=0.01 """
    print("Testing basic LeakyReLU forward pass")
    
    leakyrelu = LeakyReLU()  # Default alpha = 0.01
    print(f"LeakyReLU instance: {leakyrelu}")
    
    # Test with various values
    x_data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    x = Tensor(x_data)
    out = leakyrelu(x)
    
    # Expected result: x if x > 0, alpha * x otherwise
    alpha = 0.01
    expected = np.where(x_data > 0, x_data, alpha * x_data)
    
    print(f"Input: {x_data}")
    print(f"Output: {out.data}")
    print(f"Expected (alpha={alpha}): {expected}")
    
    assert np.allclose(out.data, expected), f"Expected {expected}, got {out.data}"
    assert out._op == "LeakyReLU", f"Expected op 'LeakyReLU', got {out._op}"
    assert x in out._prev, "Input should be in computation graph"
    print("✓ Forward pass correct")
    print("✓ Computation graph built correctly")
    return True

def test_leakyrelu_forward_custom_alpha():
    """ Test 2: LeakyReLU forward pass with custom alpha """
    print("Testing LeakyReLU forward pass with custom alpha=0.1")
    
    alpha = 0.1
    leakyrelu = LeakyReLU(alpha=alpha)
    print(f"LeakyReLU instance with alpha={alpha}: {leakyrelu}")
    
    x_data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    x = Tensor(x_data)
    out = leakyrelu(x)
    
    # Expected result: x if x > 0, alpha * x otherwise
    expected = np.where(x_data > 0, x_data, alpha * x_data)
    
    print(f"Input: {x_data}")
    print(f"Output: {out.data}")
    print(f"Expected (alpha={alpha}): {expected}")
    
    assert np.allclose(out.data, expected), f"Expected {expected}, got {out.data}"
    print("✓ Custom alpha forward pass correct")
    return True

def test_leakyrelu_forward_2d():
    """ Test 3: LeakyReLU forward pass with 2D input """
    print("Testing LeakyReLU forward pass with 2D input...")
    leakyrelu = LeakyReLU(alpha=0.01)
    
    x_data = np.array([
        [-1.0, -0.5, 0.0, 0.5],
        [1.0, 1.5, -2.0, -1.0],
        [0.0, -3.0, 4.0, 5.0]
    ])
    
    x = Tensor(x_data)
    out = leakyrelu(x)
    expected = np.where(x_data > 0, x_data, 0.01 * x_data)
    
    print(f"Input shape: {x_data.shape}")
    print(f"Output shape: {out.data.shape}")
    print(f"Sample output:\n{out.data}")
    
    assert out.data.shape == x_data.shape, f"Shape mismatch: {out.data.shape} vs {x_data.shape}"
    assert np.allclose(out.data, expected), "Output doesn't match expected"
    print("✓ 2D forward pass correct")
    return True

def test_leakyrelu_backward_basic():
    """ Test 4: Basic LeakyReLU backward pass """
    print("Testing LeakyReLU backward pass with basic values...")
    
    alpha = 0.01
    leakyrelu = LeakyReLU(alpha=alpha)
    
    x = Tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    out = leakyrelu(x)
    
    out.grad = np.ones_like(out.data)
    out._backward()
    
    # Expected gradient: 1 where x > 0, alpha elsewhere
    expected_grad = np.where(x.data > 0, 1.0, alpha)
    
    print(f"Input: {x.data}")
    print(f"Output: {out.data}")
    print(f"Computed gradients: {x.grad}")
    print(f"Expected gradients (alpha={alpha}): {expected_grad}")
    
    assert np.allclose(x.grad, expected_grad), f"Expected {expected_grad}, got {x.grad}"
    print("✓ Backward pass correct")
    
    return True

def test_leakyrelu_backward_custom_alpha():
    """ Test 5: LeakyReLU backward pass with custom alpha """
    print("Testing LeakyReLU backward pass with custom alpha=0.2...")
    
    alpha = 0.2
    leakyrelu = LeakyReLU(alpha=alpha)
    
    x = Tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    out = leakyrelu(x)
    
    out.grad = np.ones_like(out.data)
    out._backward()
    
    # Expected gradient: 1 where x > 0, alpha elsewhere
    expected_grad = np.where(x.data > 0, 1.0, alpha)
    
    print(f"Input: {x.data}")
    print(f"Output: {out.data}")
    print(f"Computed gradients: {x.grad}")
    print(f"Expected gradients (alpha={alpha}): {expected_grad}")
    
    assert np.allclose(x.grad, expected_grad), f"Expected {expected_grad}, got {x.grad}"
    print("✓ Custom alpha backward pass correct")
    
    return True

def test_leakyrelu_backward_with_loss():
    """ Test 6: LeakyReLU backward pass with loss.backward() """
    print("Testing LeakyReLU backward pass with loss.backward()...")
    
    alpha = 0.01
    leakyrelu = LeakyReLU(alpha=alpha)
    
    x = Parameter(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    out = leakyrelu(x)
    loss = out.sum()
    loss.backward()
    
    # Expected gradient: 1 where x > 0, alpha elsewhere
    expected_grad = np.where(x.data > 0, 1.0, alpha)
    
    print(f"Input: {x.data}")
    print(f"Output: {out.data}")
    print(f"Loss: {loss.data}")
    print(f"Computed gradients: {x.grad}")
    print(f"Expected gradients: {expected_grad}")
    
    assert np.allclose(x.grad, expected_grad), f"Expected {expected_grad}, got {x.grad}"
    print("✓ Backward pass with loss.backward() correct")
    return True

def test_leakyrelu_backward_2d():
    """ Test 7: LeakyReLU backward pass with 2D input """
    print("Testing LeakyReLU backward pass with 2D input...")
    
    alpha = 0.01
    leakyrelu = LeakyReLU(alpha=alpha)
    
    x_data = np.array([
        [-1.0, -0.5, 0.0, 0.5],
        [1.0, 1.5, -2.0, -1.0],
        [0.0, -3.0, 4.0, 5.0]
    ])
    x = Parameter(x_data)
    out = leakyrelu(x)
    loss = out.sum()
    loss.backward()
    
    # Expected gradient: 1 where x > 0, alpha elsewhere
    expected_grad = np.where(x_data > 0, 1.0, alpha)
    
    print(f"Input shape: {x_data.shape}")
    print(f"Output shape: {out.data.shape}")
    print(f"Computed gradients shape: {x.grad.shape}")
    print(f"Sample gradients:\n{x.grad}")
    
    assert x.grad.shape == x_data.shape, f"Gradient shape mismatch: {x.grad.shape} vs {x_data.shape}"
    assert np.allclose(x.grad, expected_grad), "Gradients don't match expected"
    print("✓ 2D backward pass correct")
    
    return True

def test_leakyrelu_computation_graph():
    """ Test 8: Verify computation graph is built correctly """
    print("Testing LeakyReLU computation graph...")
    
    leakyrelu = LeakyReLU()
    
    x = Tensor(np.array([-1.0, 0.0, 1.0]))
    y = Tensor(np.array([2.0, 2.0, 2.0]))
    
    out = leakyrelu(x) + y
    print(f"Input x: {x.data}")
    print(f"Input y: {y.data}")
    print(f"Output: {out.data}")
    print(f"Output operation: {out._op}")
    print(f"Output parents: {[id(p) for p in out._prev]}")
    
    assert out._op == '+', "Output should be from addition"
    
    # Find the LeakyReLU node in the graph
    leakyrelu_node = None
    for child in out._prev:
        if child._op == 'LeakyReLU':
            leakyrelu_node = child
            break
    
    assert leakyrelu_node is not None, "LeakyReLU node not found in computation graph"
    assert x in leakyrelu_node._prev, "Input x should be parent of LeakyReLU node"
    print("✓ Computation graph built correctly")
    
    return True

def test_leakyrelu_edge_cases():
    """ Test 9: LeakyReLU edge cases (large values) """
    print("Testing LeakyReLU edge cases...")
    
    alpha = 0.01
    leakyrelu = LeakyReLU(alpha=alpha)
    
    # Test with very large positive and negative values
    x_large = Parameter(np.array([100.0, -100.0, 1000.0, -1000.0]))
    out_large = leakyrelu(x_large)
    loss_large = out_large.sum()
    loss_large.backward()
    
    print(f"Large values input: {x_large.data}")
    print(f"Output: {out_large.data}")
    print(f"Gradients: {x_large.grad}")
    
    # For large positive x, output ≈ x, gradient ≈ 1
    # For large negative x, output ≈ alpha*x, gradient ≈ alpha
    expected_output = np.array([100.0, -1.0, 1000.0, -10.0])
    expected_grad = np.array([1.0, alpha, 1.0, alpha])
    
    assert np.allclose(out_large.data, expected_output), "Large value output incorrect"
    assert np.allclose(x_large.grad, expected_grad), "Large value gradients incorrect"
    print("✓ Edge cases handled correctly")
    
    return True

def test_leakyrelu_properties():
    """ Test 10: LeakyReLU special properties """
    print("Testing LeakyReLU properties...")
    
    alpha = 0.05
    leakyrelu = LeakyReLU(alpha=alpha)
    
    # Test that negative values are scaled by alpha
    x_neg = Parameter(np.array([-2.0, -1.0]))
    out_neg = leakyrelu(x_neg)
    expected_neg = alpha * x_neg.data
    assert np.allclose(out_neg.data, expected_neg), f"Negative values should be scaled by {alpha}"
    print(f"✓ Negative values scaled by alpha={alpha}")
    
    # Test that positive values pass through unchanged
    x_pos = Parameter(np.array([1.0, 2.0]))
    out_pos = leakyrelu(x_pos)
    expected_pos = x_pos.data
    assert np.allclose(out_pos.data, expected_pos), "Positive values should pass through unchanged"
    print("✓ Positive values unchanged")
    
    # Test gradient for negative values
    x_neg_grad = Parameter(np.array([-1.0]))
    out_neg_grad = leakyrelu(x_neg_grad)
    loss_neg = out_neg_grad.sum()
    loss_neg.backward()
    assert np.allclose(x_neg_grad.grad, alpha), f"Gradient for negative should be {alpha}"
    print(f"✓ Gradient for negative values = {alpha}")
    
    return True

def test_leakyrelu_different_alphas():
    """ Test 11: Different alpha values """
    print("Testing LeakyReLU with different alpha values...")
    
    alphas = [0.01, 0.1, 0.2, 0.5]
    x_data = np.array([-2.0, -1.0, 1.0, 2.0])
    
    for alpha in alphas:
        leakyrelu = LeakyReLU(alpha=alpha)
        x = Tensor(x_data)
        out = leakyrelu(x)
        
        expected = np.where(x_data > 0, x_data, alpha * x_data)
        
        print(f"Alpha={alpha}: {out.data}")
        assert np.allclose(out.data, expected), f"Failed for alpha={alpha}"
    
    print("✓ All alpha values work correctly")
    return True

def test_leakyrelu_multiple_uses():
    """ Test 12: Same LeakyReLU instance used multiple times """
    print("Testing LeakyReLU instance used multiple times...")
    
    alpha = 0.01
    leakyrelu = LeakyReLU(alpha=alpha)
    
    # Create different inputs
    x1 = Parameter(np.array([-1.0, 0.0, 1.0]))
    x2 = Parameter(np.array([-2.0, 2.0, -3.0]))
    
    print(f"x1 input: {x1.data}")
    print(f"x2 input: {x2.data}")
    
    # Use same LeakyReLU instance twice
    out1 = leakyrelu(x1)
    out2 = leakyrelu(x2)
    
    print(f"out1 (leakyrelu of x1): {out1.data}")
    print(f"out2 (leakyrelu of x2): {out2.data}")
    
    # Calculate expected values manually
    expected_out1 = np.where(x1.data > 0, x1.data, alpha * x1.data)
    expected_out2 = np.where(x2.data > 0, x2.data, alpha * x2.data)
    
    print(f"Expected out1: {expected_out1}")
    print(f"Expected out2: {expected_out2}")
    
    # Verify forward passes are correct and different
    assert np.allclose(out1.data, expected_out1), "out1 forward pass incorrect"
    assert np.allclose(out2.data, expected_out2), "out2 forward pass incorrect"
    assert not np.allclose(out1.data, out2.data), "Outputs should be different for different inputs"
    print("✓ Forward passes correct and different")
    
    # Create separate losses and sum them
    loss1 = out1.sum()
    loss2 = out2.sum()
    loss = loss1 + loss2
    
    print(f"loss1: {loss1.data}")
    print(f"loss2: {loss2.data}")
    print(f"total loss: {loss.data}")
    
    # Backward pass
    loss.backward()
    
    print(f"x1 gradients: {x1.grad}")
    print(f"x2 gradients: {x2.grad}")
    
    # Expected gradients: 1 where x > 0, alpha elsewhere
    expected_grad_x1 = np.where(x1.data > 0, 1.0, alpha)
    expected_grad_x2 = np.where(x2.data > 0, 1.0, alpha)
    
    print(f"Expected x1 gradients: {expected_grad_x1}")
    print(f"Expected x2 gradients: {expected_grad_x2}")
    
    # Check if gradients are correct and different from each other
    grad_diff = not np.allclose(x1.grad, x2.grad)
    x1_correct = np.allclose(x1.grad, expected_grad_x1)
    x2_correct = np.allclose(x2.grad, expected_grad_x2)
    
    print(f"Gradients different? {grad_diff}")
    print(f"x1 correct? {x1_correct}")
    print(f"x2 correct? {x2_correct}")
    
    assert grad_diff, "Gradients should be different for different inputs"
    assert x1_correct, f"x1 gradients: got {x1.grad}, expected {expected_grad_x1}"
    assert x2_correct, f"x2 gradients: got {x2.grad}, expected {expected_grad_x2}"
    
    print("✓ Multiple uses of same LeakyReLU instance works correctly")
    
    return True

def main():
    tests = [
        ('Forward Pass Basic', test_leakyrelu_forward_basic),
        ('Forward Pass Custom Alpha', test_leakyrelu_forward_custom_alpha),
        ('Forward Pass 2D', test_leakyrelu_forward_2d),
        ('Backward Pass Basic', test_leakyrelu_backward_basic),
        ('Backward Pass Custom Alpha', test_leakyrelu_backward_custom_alpha),
        ('Backward with Loss', test_leakyrelu_backward_with_loss),
        ('Backward Pass 2D', test_leakyrelu_backward_2d),
        ('Computation Graph', test_leakyrelu_computation_graph),
        ('Edge Cases', test_leakyrelu_edge_cases),
        ('LeakyReLU Properties', test_leakyrelu_properties),
        ('Different Alphas', test_leakyrelu_different_alphas),
        ('Multiple Uses', test_leakyrelu_multiple_uses),
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
    print("TEST SUMMARY - LEAKYRELU")
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
        print("\n🎉 ALL LEAKYRELU TESTS PASSED 🎉")
        print("LeakyReLU implementation is working correctly!")
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