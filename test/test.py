import numpy as np
import sys
import traceback

def pause_on_error():
    print("\n" + "=" * 60)
    input("Press Enter to continue after error...")
    

try:
    from ann_engine.core import Tensor, Parameter
    from ann_engine.layers import Sigmoid  # Changed to Sigmoid
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
    

def test_sigmoid_forward_basic():
    """ Test 1: Basic Sigmoid forward pass """
    print("Testing basic Sigmoid forward pass")
    
    sigmoid = Sigmoid()
    print(f"Sigmoid instance: {sigmoid}")
    
    # Test with various values
    x_data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    x = Tensor(x_data)
    out = sigmoid(x)
    
    # Expected result: 1 / (1 + exp(-x))
    expected = 1 / (1 + np.exp(-x_data))
    
    print(f"Input: {x_data}")
    print(f"Output: {out.data}")
    print(f"Expected: {expected}")
    
    assert np.allclose(out.data, expected), f"Expected {expected}, got {out.data}"
    assert out._op == "Sigmoid", f"Expected op 'Sigmoid', got {out._op}"
    assert x in out._prev, "Input should be in computation graph"
    print("✓ Forward pass correct")
    print("✓ Computation graph built correctly")
    return True

def test_sigmoid_forward_2d():
    """ Test 2: Sigmoid forward pass with 2D input """
    print("Testing Sigmoid forward pass with 2D input...")
    sigmoid = Sigmoid()
    
    x_data = np.array([
        [-1.0, -0.5, 0.0, 0.5],
        [1.0, 1.5, -2.0, -1.0],
        [0.0, -3.0, 4.0, 5.0]
    ])
    
    x = Tensor(x_data)
    out = sigmoid(x)
    expected = 1 / (1 + np.exp(-x_data))
    
    print(f"Input shape: {x_data.shape}")
    print(f"Output shape: {out.data.shape}")
    print(f"Sample output:\n{out.data}")
    
    assert out.data.shape == x_data.shape, f"Shape mismatch: {out.data.shape} vs {x_data.shape}"
    assert np.allclose(out.data, expected), "Output doesn't match expected"
    print("✓ 2D forward pass correct")
    return True

def test_sigmoid_backward_basic():
    """ Test 3: Basic Sigmoid backward pass """
    print("Testing Sigmoid backward pass with basic values...")
    sigmoid = Sigmoid()
    
    x = Tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    out = sigmoid(x)
    
    # Sigmoid output for gradient calculation
    sigmoid_output = out.data
    
    out.grad = np.ones_like(out.data)
    out._backward()
    
    # Expected gradient: sigmoid(x) * (1 - sigmoid(x))
    expected_grad = sigmoid_output * (1 - sigmoid_output)
    
    print(f"Input: {x.data}")
    print(f"Sigmoid output: {sigmoid_output}")
    print(f"Computed gradients: {x.grad}")
    print(f"Expected gradients: {expected_grad}")
    
    assert np.allclose(x.grad, expected_grad), f"Expected {expected_grad}, got {x.grad}"
    print("✓ Backward pass correct")
    
    return True

def test_sigmoid_backward_with_loss():
    """ Test 4: Sigmoid backward pass with loss.backward() """
    print("Testing Sigmoid backward pass with loss.backward()...")
    sigmoid = Sigmoid()
    
    x = Parameter(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    out = sigmoid(x)
    loss = out.sum()
    loss.backward()
    
    # Expected gradient: sigmoid(x) * (1 - sigmoid(x))
    sigmoid_output = out.data
    expected_grad = sigmoid_output * (1 - sigmoid_output)
    
    print(f"Input: {x.data}")
    print(f"Sigmoid output: {sigmoid_output}")
    print(f"Loss: {loss.data}")
    print(f"Computed gradients: {x.grad}")
    print(f"Expected gradients: {expected_grad}")
    
    assert np.allclose(x.grad, expected_grad), f"Expected {expected_grad}, got {x.grad}"
    print("✓ Backward pass with loss.backward() correct")
    return True

def test_sigmoid_backward_2d():
    """ Test 5: Sigmoid backward pass with 2D input """
    print("Testing Sigmoid backward pass with 2D input...")
    sigmoid = Sigmoid()
    
    x_data = np.array([
        [-1.0, -0.5, 0.0, 0.5],
        [1.0, 1.5, -2.0, -1.0],
        [0.0, -3.0, 4.0, 5.0]
    ])
    x = Parameter(x_data)
    out = sigmoid(x)
    loss = out.sum()
    loss.backward()
    
    # Expected gradient: sigmoid(x) * (1 - sigmoid(x))
    sigmoid_output = out.data
    expected_grad = sigmoid_output * (1 - sigmoid_output)
    
    print(f"Input shape: {x_data.shape}")
    print(f"Output shape: {out.data.shape}")
    print(f"Computed gradients shape: {x.grad.shape}")
    print(f"Sample gradients:\n{x.grad}")
    
    assert x.grad.shape == x_data.shape, f"Gradient shape mismatch: {x.grad.shape} vs {x_data.shape}"
    assert np.allclose(x.grad, expected_grad), "Gradients don't match expected"
    print("✓ 2D backward pass correct")
    
    return True

def test_sigmoid_computation_graph():
    """ Test 6: Verify computation graph is built correctly """
    print("Testing Sigmoid computation graph...")
    
    sigmoid = Sigmoid()
    
    x = Tensor(np.array([-1.0, 0.0, 1.0]))
    y = Tensor(np.array([2.0, 2.0, 2.0]))
    
    out = sigmoid(x) + y
    print(f"Input x: {x.data}")
    print(f"Input y: {y.data}")
    print(f"Output: {out.data}")
    print(f"Output operation: {out._op}")
    print(f"Output parents: {[id(p) for p in out._prev]}")
    
    assert out._op == '+', "Output should be from addition"
    
    # Find the Sigmoid node in the graph
    sigmoid_node = None
    for child in out._prev:
        if child._op == 'Sigmoid':
            sigmoid_node = child
            break
    
    assert sigmoid_node is not None, "Sigmoid node not found in computation graph"
    assert x in sigmoid_node._prev, "Input x should be parent of Sigmoid node"
    print("✓ Computation graph built correctly")
    
    return True

def test_sigmoid_edge_cases():
    """ Test 7: Sigmoid edge cases (large values) """
    print("Testing Sigmoid edge cases...")
    
    sigmoid = Sigmoid()
    
    # Test with very large positive and negative values
    x_large = Parameter(np.array([100.0, -100.0, 50.0, -50.0]))
    out_large = sigmoid(x_large)
    loss_large = out_large.sum()
    loss_large.backward()
    
    print(f"Large values input: {x_large.data}")
    print(f"Output: {out_large.data}")
    print(f"Gradients: {x_large.grad}")
    
    # For large positive x, sigmoid ≈ 1
    # For large negative x, sigmoid ≈ 0
    expected_output = np.array([1.0, 0.0, 1.0, 0.0])
    # For large positive x, gradient ≈ 0
    # For large negative x, gradient ≈ 0
    expected_grad = np.array([0.0, 0.0, 0.0, 0.0])
    
    assert np.allclose(out_large.data, expected_output, rtol=1e-3), "Large value output incorrect"
    assert np.allclose(x_large.grad, expected_grad, rtol=1e-3), "Large value gradients incorrect"
    print("✓ Edge cases handled correctly")
    
    return True

def test_sigmoid_properties():
    """ Test 8: Sigmoid special properties """
    print("Testing Sigmoid properties...")
    
    sigmoid = Sigmoid()
    
    x = Parameter(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    out = sigmoid(x)
    
    # Property 1: Sigmoid output is between 0 and 1
    assert np.all(out.data >= 0) and np.all(out.data <= 1), "Sigmoid output should be in [0,1]"
    print("✓ Property 1: Output in [0,1]")
    
    # Property 2: Sigmoid(0) = 0.5
    x_zero = Parameter(np.array([0.0]))
    out_zero = sigmoid(x_zero)
    assert np.allclose(out_zero.data, 0.5), f"Sigmoid(0) should be 0.5, got {out_zero.data}"
    print("✓ Property 2: Sigmoid(0) = 0.5")
    
    # Property 3: Sigmoid(-x) = 1 - Sigmoid(x)
    x_pos = Parameter(np.array([2.0]))
    x_neg = Parameter(np.array([-2.0]))
    out_pos = sigmoid(x_pos)
    out_neg = sigmoid(x_neg)
    assert np.allclose(out_pos.data, 1 - out_neg.data), f"Sigmoid(-x) should equal 1 - Sigmoid(x)"
    print("✓ Property 3: Sigmoid(-x) = 1 - Sigmoid(x)")
    
    return True

def test_sigmoid_multiple_uses():
    """ Test 9: Same Sigmoid instance used multiple times """
    print("Testing Sigmoid instance used multiple times...")
    
    sigmoid = Sigmoid()
    
    x1 = Parameter(np.array([-1.0, 0.0, 1.0]))
    x2 = Parameter(np.array([-2.0, 2.0, -3.0]))
    
    # Use same Sigmoid instance twice
    out1 = sigmoid(x1)
    out2 = sigmoid(x2)
    
    loss = out1.sum() + out2.sum()
    loss.backward()
    
    print(f"x1 gradients: {x1.grad}")
    print(f"x2 gradients: {x2.grad}")
    
    # Expected gradients: sigmoid(x) * (1 - sigmoid(x))
    expected_grad_x1 = 1/(1+np.exp(-x1.data)) * (1 - 1/(1+np.exp(-x1.data)))
    expected_grad_x2 = 1/(1+np.exp(-x2.data)) * (1 - 1/(1+np.exp(-x2.data)))
    
    assert np.allclose(x1.grad, expected_grad_x1), "x1 gradients incorrect"
    assert np.allclose(x2.grad, expected_grad_x2), "x2 gradients incorrect"
    print("✓ Multiple uses of same Sigmoid instance works")
    
    return True

def main():
    tests = [
        ('Forward Pass Basic', test_sigmoid_forward_basic),
        ('Forward Pass 2D', test_sigmoid_forward_2d),
        ('Backward Pass Basic', test_sigmoid_backward_basic),
        ('Backward with Loss', test_sigmoid_backward_with_loss),
        ('Backward Pass 2D', test_sigmoid_backward_2d),
        ('Computation Graph', test_sigmoid_computation_graph),
        ('Edge Cases', test_sigmoid_edge_cases),
        ('Sigmoid Properties', test_sigmoid_properties),
        ('Multiple Uses', test_sigmoid_multiple_uses),
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
    print("TEST SUMMARY - SIGMOID")
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
        print("\n🎉 ALL SIGMOID TESTS PASSED 🎉")
        print("Sigmoid implementation is working correctly!")
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