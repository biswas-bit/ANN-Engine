import numpy as np
import sys
import traceback

def pause_on_error():
    print("\n" + "=" * 60)
    input("Press Enter to continue after error...")
    

try:
    from ann_engine.core import Tensor, Parameter
    from ann_engine.layers import Tanh  # Changed to Tanh
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
    

def test_tanh_forward_basic():
    """ Test 1: Basic Tanh forward pass """
    print("Testing basic Tanh forward pass")
    
    tanh = Tanh()
    print(f"Tanh instance: {tanh}")
    
    # Test with various values
    x_data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    x = Tensor(x_data)
    out = tanh(x)
    
    # Expected result: tanh(x)
    expected = np.tanh(x_data)
    
    print(f"Input: {x_data}")
    print(f"Output: {out.data}")
    print(f"Expected: {expected}")
    
    assert np.allclose(out.data, expected), f"Expected {expected}, got {out.data}"
    assert out._op == "Tanh", f"Expected op 'Tanh', got {out._op}"
    assert x in out._prev, "Input should be in computation graph"
    print("✓ Forward pass correct")
    print("✓ Computation graph built correctly")
    return True

def test_tanh_forward_2d():
    """ Test 2: Tanh forward pass with 2D input """
    print("Testing Tanh forward pass with 2D input...")
    tanh = Tanh()
    
    x_data = np.array([
        [-1.0, -0.5, 0.0, 0.5],
        [1.0, 1.5, -2.0, -1.0],
        [0.0, -3.0, 4.0, 5.0]
    ])
    
    x = Tensor(x_data)
    out = tanh(x)
    expected = np.tanh(x_data)
    
    print(f"Input shape: {x_data.shape}")
    print(f"Output shape: {out.data.shape}")
    print(f"Sample output:\n{out.data}")
    
    assert out.data.shape == x_data.shape, f"Shape mismatch: {out.data.shape} vs {x_data.shape}"
    assert np.allclose(out.data, expected), "Output doesn't match expected"
    print("✓ 2D forward pass correct")
    return True

def test_tanh_backward_basic():
    """ Test 3: Basic Tanh backward pass """
    print("Testing Tanh backward pass with basic values...")
    tanh = Tanh()
    
    x = Tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    out = tanh(x)
    
    # Tanh output for gradient calculation
    tanh_output = out.data
    
    out.grad = np.ones_like(out.data)
    out._backward()
    
    # Expected gradient: 1 - tanh^2(x)
    expected_grad = 1 - tanh_output ** 2
    
    print(f"Input: {x.data}")
    print(f"Tanh output: {tanh_output}")
    print(f"Computed gradients: {x.grad}")
    print(f"Expected gradients: {expected_grad}")
    
    assert np.allclose(x.grad, expected_grad), f"Expected {expected_grad}, got {x.grad}"
    print("✓ Backward pass correct")
    
    return True

def test_tanh_backward_with_loss():
    """ Test 4: Tanh backward pass with loss.backward() """
    print("Testing Tanh backward pass with loss.backward()...")
    tanh = Tanh()
    
    x = Parameter(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    out = tanh(x)
    loss = out.sum()
    loss.backward()
    
    # Expected gradient: 1 - tanh^2(x)
    tanh_output = out.data
    expected_grad = 1 - tanh_output ** 2
    
    print(f"Input: {x.data}")
    print(f"Tanh output: {tanh_output}")
    print(f"Loss: {loss.data}")
    print(f"Computed gradients: {x.grad}")
    print(f"Expected gradients: {expected_grad}")
    
    assert np.allclose(x.grad, expected_grad), f"Expected {expected_grad}, got {x.grad}"
    print("✓ Backward pass with loss.backward() correct")
    return True

def test_tanh_backward_2d():
    """ Test 5: Tanh backward pass with 2D input """
    print("Testing Tanh backward pass with 2D input...")
    tanh = Tanh()
    
    x_data = np.array([
        [-1.0, -0.5, 0.0, 0.5],
        [1.0, 1.5, -2.0, -1.0],
        [0.0, -3.0, 4.0, 5.0]
    ])
    x = Parameter(x_data)
    out = tanh(x)
    loss = out.sum()
    loss.backward()
    
    # Expected gradient: 1 - tanh^2(x)
    tanh_output = out.data
    expected_grad = 1 - tanh_output ** 2
    
    print(f"Input shape: {x_data.shape}")
    print(f"Output shape: {out.data.shape}")
    print(f"Computed gradients shape: {x.grad.shape}")
    print(f"Sample gradients:\n{x.grad}")
    
    assert x.grad.shape == x_data.shape, f"Gradient shape mismatch: {x.grad.shape} vs {x_data.shape}"
    assert np.allclose(x.grad, expected_grad), "Gradients don't match expected"
    print("✓ 2D backward pass correct")
    
    return True

def test_tanh_computation_graph():
    """ Test 6: Verify computation graph is built correctly """
    print("Testing Tanh computation graph...")
    
    tanh = Tanh()
    
    x = Tensor(np.array([-1.0, 0.0, 1.0]))
    y = Tensor(np.array([2.0, 2.0, 2.0]))
    
    out = tanh(x) + y
    print(f"Input x: {x.data}")
    print(f"Input y: {y.data}")
    print(f"Output: {out.data}")
    print(f"Output operation: {out._op}")
    print(f"Output parents: {[id(p) for p in out._prev]}")
    
    assert out._op == '+', "Output should be from addition"
    
    # Find the Tanh node in the graph
    tanh_node = None
    for child in out._prev:
        if child._op == 'Tanh':
            tanh_node = child
            break
    
    assert tanh_node is not None, "Tanh node not found in computation graph"
    assert x in tanh_node._prev, "Input x should be parent of Tanh node"
    print("✓ Computation graph built correctly")
    
    return True

def test_tanh_edge_cases():
    """ Test 7: Tanh edge cases (large values) """
    print("Testing Tanh edge cases...")
    
    tanh = Tanh()
    
    # Test with very large positive and negative values
    x_large = Parameter(np.array([100.0, -100.0, 50.0, -50.0]))
    out_large = tanh(x_large)
    loss_large = out_large.sum()
    loss_large.backward()
    
    print(f"Large values input: {x_large.data}")
    print(f"Output: {out_large.data}")
    print(f"Gradients: {x_large.grad}")
    
    # For large positive x, tanh ≈ 1
    # For large negative x, tanh ≈ -1
    expected_output = np.array([1.0, -1.0, 1.0, -1.0])
    # For large |x|, gradient ≈ 0
    expected_grad = np.array([0.0, 0.0, 0.0, 0.0])
    
    assert np.allclose(out_large.data, expected_output, rtol=1e-3), "Large value output incorrect"
    assert np.allclose(x_large.grad, expected_grad, rtol=1e-3), "Large value gradients incorrect"
    print("✓ Edge cases handled correctly")
    
    return True

def test_tanh_properties():
    """ Test 8: Tanh special properties """
    print("Testing Tanh properties...")
    
    tanh = Tanh()
    
    x = Parameter(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    out = tanh(x)
    
    # Property 1: Tanh output is between -1 and 1
    assert np.all(out.data >= -1) and np.all(out.data <= 1), "Tanh output should be in [-1, 1]"
    print("✓ Property 1: Output in [-1, 1]")
    
    # Property 2: Tanh(0) = 0
    x_zero = Parameter(np.array([0.0]))
    out_zero = tanh(x_zero)
    assert np.allclose(out_zero.data, 0.0), f"Tanh(0) should be 0, got {out_zero.data}"
    print("✓ Property 2: Tanh(0) = 0")
    
    # Property 3: Tanh is odd function: tanh(-x) = -tanh(x)
    x_pos = Parameter(np.array([2.0]))
    x_neg = Parameter(np.array([-2.0]))
    out_pos = tanh(x_pos)
    out_neg = tanh(x_neg)
    assert np.allclose(out_pos.data, -out_neg.data), f"Tanh(-x) should equal -Tanh(x)"
    print("✓ Property 3: Tanh is odd function")
    
    # Property 4: Gradient at 0 is 1
    x_zero_grad = Parameter(np.array([0.0]))
    out_zero_grad = tanh(x_zero_grad)
    loss_zero = out_zero_grad.sum()
    loss_zero.backward()
    assert np.allclose(x_zero_grad.grad, 1.0), f"Gradient at 0 should be 1, got {x_zero_grad.grad}"
    print("✓ Property 4: Gradient at 0 = 1")
    
    return True

def test_tanh_multiple_uses():
    """ Test 9: Same Tanh instance used multiple times """
    print("Testing Tanh instance used multiple times...")
    
    tanh = Tanh()
    
    # Create different inputs
    x1 = Parameter(np.array([-1.0, 0.0, 1.0]))
    x2 = Parameter(np.array([-2.0, 2.0, -3.0]))
    
    print(f"x1 input: {x1.data}")
    print(f"x2 input: {x2.data}")
    
    # Use same Tanh instance twice
    out1 = tanh(x1)
    out2 = tanh(x2)
    
    print(f"out1 (tanh of x1): {out1.data}")
    print(f"out2 (tanh of x2): {out2.data}")
    
    # Calculate expected tanh values manually
    expected_out1 = np.tanh(x1.data)
    expected_out2 = np.tanh(x2.data)
    
    print(f"Expected out1: {expected_out1}")
    print(f"Expected out2: {expected_out2}")
    
    # Verify forward passes are different
    assert not np.allclose(out1.data, out2.data), "Outputs should be different for different inputs"
    
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
    
    # Expected gradients: 1 - tanh^2(x)
    expected_grad_x1 = 1 - expected_out1 ** 2
    expected_grad_x2 = 1 - expected_out2 ** 2
    
    print(f"Expected x1 gradients: {expected_grad_x1}")
    print(f"Expected x2 gradients: {expected_grad_x2}")
    
    # Check if gradients are correct and different from each other
    assert not np.allclose(x1.grad, x2.grad), "Gradients should be different for different inputs"
    assert np.allclose(x1.grad, expected_grad_x1), f"x1 gradients: got {x1.grad}, expected {expected_grad_x1}"
    assert np.allclose(x2.grad, expected_grad_x2), f"x2 gradients: got {x2.grad}, expected {expected_grad_x2}"
    
    print("✓ Multiple uses of same Tanh instance works correctly")
    
    return True

def main():
    tests = [
        ('Forward Pass Basic', test_tanh_forward_basic),
        ('Forward Pass 2D', test_tanh_forward_2d),
        ('Backward Pass Basic', test_tanh_backward_basic),
        ('Backward with Loss', test_tanh_backward_with_loss),
        ('Backward Pass 2D', test_tanh_backward_2d),
        ('Computation Graph', test_tanh_computation_graph),
        ('Edge Cases', test_tanh_edge_cases),
        ('Tanh Properties', test_tanh_properties),
        ('Multiple Uses', test_tanh_multiple_uses),
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
    print("TEST SUMMARY - TANH")
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
        print("\n🎉 ALL TANH TESTS PASSED 🎉")
        print("Tanh implementation is working correctly!")
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