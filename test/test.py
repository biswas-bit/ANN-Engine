import numpy as np
import sys
import traceback

def pause_on_error():
    print("\n" + "=" * 60)
    input("Press Enter to continue after error...")

try:
    from ann_engine.core import Tensor, Parameter
    from ann_engine.layers import Softmax
except ImportError as e:
    print(f"Failed to Import libraries : {e}")
    pause_on_error()
    sys.exit(1)

def run_test(test_name, test_func):
    print(f"\n" + "=" * 50)
    print(f"Running test : {test_name}")
    print("=" * 50)
    
    try:
        result = test_func()
        print(f"\n✓ Test Passed : {test_name}")
        return True, result
    except AssertionError as e:
        print(f"\n✗ Test Failed (Assertion Error): {test_name}")
        print(f"  {e}")
        print("\n" + "-" * 50)
        traceback.print_exc()
        pause_on_error()
        return False, None
    except Exception as e:
        print(f'\n✗ Test Failed (Exception): {test_name}')
        print(f"  Error: {e}")
        print("\n" + "-" * 50)
        traceback.print_exc()
        pause_on_error()
        return False, None


def test_softmax_forward_basic():
    """Test 1: Basic Softmax forward pass"""
    print("Testing basic Softmax forward pass...")
    
    softmax = Softmax()
    
    x_data = np.array([[1.0, 2.0, 3.0],
                       [1.0, 2.0, 3.0]])
    x = Tensor(x_data)
    out = softmax(x)
    
    # Calculate expected
    exp_x = np.exp(x_data - np.max(x_data, axis=-1, keepdims=True))
    expected = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    print(f"Input:\n{x_data}")
    print(f"Output:\n{out.data}")
    print(f"Expected:\n{expected}")
    
    assert np.allclose(out.data, expected), f"Expected {expected}, got {out.data}"
    
    # Check that each row sums to 1
    row_sums = np.sum(out.data, axis=-1)
    assert np.allclose(row_sums, 1.0), f"Rows should sum to 1, got {row_sums}"
    
    print("✓ Forward pass correct")
    print("✓ Rows sum to 1")
    return True


def test_softmax_forward_different_dim():
    """Test 2: Softmax with different dimension"""
    print("Testing Softmax with dim=0...")
    
    softmax = Softmax(dim=0)
    
    x_data = np.array([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]])
    x = Tensor(x_data)
    out = softmax(x)
    
    # Calculate expected
    exp_x = np.exp(x_data - np.max(x_data, axis=0, keepdims=True))
    expected = exp_x / np.sum(exp_x, axis=0, keepdims=True)
    
    print(f"Input:\n{x_data}")
    print(f"Output:\n{out.data}")
    print(f"Expected:\n{expected}")
    
    assert np.allclose(out.data, expected), f"Expected {expected}, got {out.data}"
    
    # Check that each column sums to 1
    col_sums = np.sum(out.data, axis=0)
    assert np.allclose(col_sums, 1.0), f"Columns should sum to 1, got {col_sums}"
    
    print("✓ Forward pass with dim=0 correct")
    return True


def test_softmax_forward_1d():
    """Test 3: Softmax with 1D input"""
    print("Testing Softmax with 1D input...")
    
    softmax = Softmax()
    
    x_data = np.array([1.0, 2.0, 3.0])
    x = Tensor(x_data)
    out = softmax(x)
    
    # Calculate expected
    exp_x = np.exp(x_data - np.max(x_data))
    expected = exp_x / np.sum(exp_x)
    
    print(f"Input: {x_data}")
    print(f"Output: {out.data}")
    print(f"Expected: {expected}")
    
    assert np.allclose(out.data, expected), f"Expected {expected}, got {out.data}"
    assert np.allclose(np.sum(out.data), 1.0), f"Should sum to 1"
    
    print("✓ 1D forward pass correct")
    return True


def test_softmax_forward_3d():
    """Test 4: Softmax with 3D input"""
    print("Testing Softmax with 3D input...")
    
    softmax = Softmax(dim=-1)
    
    x_data = np.random.randn(2, 3, 4)
    x = Tensor(x_data)
    out = softmax(x)
    
    # Calculate expected
    exp_x = np.exp(x_data - np.max(x_data, axis=-1, keepdims=True))
    expected = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    print(f"Input shape: {x_data.shape}")
    print(f"Output shape: {out.data.shape}")
    
    assert out.data.shape == x_data.shape, f"Shape mismatch"
    assert np.allclose(out.data, expected), "Output doesn't match expected"
    
    # Check sums
    sums = np.sum(out.data, axis=-1)
    assert np.allclose(sums, 1.0), "Last dimension should sum to 1"
    
    print("✓ 3D forward pass correct")
    return True


def test_softmax_numerical_stability():
    """Test 5: Numerical stability with large values"""
    print("Testing numerical stability...")
    
    softmax = Softmax()
    
    # Test with very large values
    x_large = np.array([[1000.0, 1000.0, 1000.0]])
    x = Tensor(x_large)
    out = softmax(x)
    
    expected = np.array([[1/3, 1/3, 1/3]])
    
    print(f"Large input: {x_large}")
    print(f"Output: {out.data}")
    
    assert np.allclose(out.data, expected, rtol=1e-3), f"Expected {expected}, got {out.data}"
    assert not np.any(np.isnan(out.data)), "Output contains NaN"
    assert not np.any(np.isinf(out.data)), "Output contains Inf"
    
    # Test with very different values
    x_diff = np.array([[1000.0, 0.0, -1000.0]])
    out_diff = softmax(Tensor(x_diff))
    
    print(f"Different input: {x_diff}")
    print(f"Output: {out_diff.data}")
    
    # First element should be ~1
    assert out_diff.data[0, 0] > 0.999, "First element should be near 1"
    
    print("✓ Numerical stability correct")
    return True


def test_softmax_backward_basic():
    """Test 6: Basic backward pass"""
    print("Testing backward pass...")
    
    softmax = Softmax()
    
    x_data = np.array([[1.0, 2.0, 3.0],
                       [1.0, 2.0, 3.0]])
    x = Parameter(x_data)
    out = softmax(x)
    
    # Simple upstream gradient (all ones)
    out.grad = np.ones_like(out.data)
    out.backward()
    
    print(f"Input:\n{x_data}")
    print(f"Softmax output:\n{out.data}")
    print(f"Gradients:\n{x.grad}")
    
    assert x.grad is not None, "Gradients should be computed"
    assert x.grad.shape == x_data.shape, f"Gradient shape mismatch"
    
    # For softmax with uniform upstream gradient, 
    # gradient should sum to ~0 along softmax dimension
    grad_sums = np.sum(x.grad, axis=-1)
    assert np.allclose(grad_sums, 0.0, atol=1e-6), \
        f"Gradient sums should be ~0, got {grad_sums}"
    
    print("✓ Backward pass produces correct gradients")
    return True


def test_softmax_backward_with_loss():
    """Test 7: Backward with actual loss"""
    print("Testing backward with loss...")
    
    softmax = Softmax()
    
    # Simple example
    x_data = np.array([[2.0, 1.0, 0.1],
                       [0.1, 3.0, 0.2]])
    x = Parameter(x_data)
    
    probs = softmax(x)
    
    # Simple loss: sum of probabilities
    # This creates a computation graph
    loss_value = 0.0
    for i in range(probs.data.shape[0]):
        for j in range(probs.data.shape[1]):
            loss_value += probs.data[i, j]
    
    # Create loss tensor and set gradient
    probs.grad = np.ones_like(probs.data)
    probs.backward()
    
    print(f"Input:\n{x_data}")
    print(f"Probabilities:\n{probs.data}")
    print(f"Gradients:\n{x.grad}")
    
    assert x.grad is not None, "Gradients should exist"
    assert x.grad.shape == x_data.shape, "Shape mismatch"
    
    print("✓ Backward with loss works")
    return True


def test_softmax_properties():
    """Test 8: Softmax mathematical properties"""
    print("Testing Softmax properties...")
    
    softmax = Softmax()
    
    x_data = np.array([[1.0, 2.0, 3.0]])
    x = Tensor(x_data)
    out = softmax(x)
    
    # Property 1: All outputs in [0, 1]
    assert np.all(out.data >= 0) and np.all(out.data <= 1), \
        "Outputs should be in [0, 1]"
    print("✓ Property 1: Outputs in [0, 1]")
    
    # Property 2: Outputs sum to 1
    assert np.allclose(np.sum(out.data, axis=-1), 1.0), \
        "Outputs should sum to 1"
    print("✓ Property 2: Outputs sum to 1")
    
    # Property 3: Invariant to constant shifts
    x_shifted = x_data + 100
    x_shift = Tensor(x_shifted)
    out_shift = softmax(x_shift)
    
    assert np.allclose(out.data, out_shift.data), \
        "Softmax should be invariant to constant shifts"
    print("✓ Property 3: Invariant to constant shifts")
    
    # Property 4: Monotonicity
    assert out.data[0, 2] > out.data[0, 1] > out.data[0, 0], \
        "Higher inputs should give higher probabilities"
    print("✓ Property 4: Monotonicity preserved")
    
    return True


def test_softmax_computation_graph():
    """Test 9: Computation graph structure"""
    print("Testing computation graph...")
    
    softmax = Softmax()
    
    x = Tensor(np.array([[1.0, 2.0, 3.0]]))
    out = softmax(x)
    
    print(f"Input: {x.data}")
    print(f"Output: {out.data}")
    print(f"Output op: {out._op}")
    
    assert out._op == 'Softmax', f"Expected op 'Softmax', got {out._op}"
    assert x in out._prev, "Input should be in computation graph"
    
    print("✓ Computation graph built correctly")
    return True


def test_softmax_multiple_calls():
    """Test 10: Multiple calls to same Softmax instance"""
    print("Testing multiple calls...")
    
    softmax = Softmax()
    
    x1 = Parameter(np.array([[1.0, 2.0, 3.0]]))
    x2 = Parameter(np.array([[4.0, 5.0, 6.0]]))
    
    out1 = softmax(x1)
    out2 = softmax(x2)
    
    print(f"x1: {x1.data} -> out1: {out1.data}")
    print(f"x2: {x2.data} -> out2: {out2.data}")
    
    # Calculate expected
    exp_x1 = np.exp(x1.data - np.max(x1.data))
    exp_x2 = np.exp(x2.data - np.max(x2.data))
    expected_out1 = exp_x1 / np.sum(exp_x1)
    expected_out2 = exp_x2 / np.sum(exp_x2)
    
    assert np.allclose(out1.data, expected_out1), "out1 incorrect"
    assert np.allclose(out2.data, expected_out2), "out2 incorrect"
    assert not np.allclose(out1.data, out2.data), "Outputs should be different"
    
    # Test backward separately
    out1.grad = np.ones_like(out1.data)
    out1.backward()
    
    out2.grad = np.ones_like(out2.data)
    out2.backward()
    
    assert x1.grad is not None, "x1 should have gradients"
    assert x2.grad is not None, "x2 should have gradients"
    
    print("✓ Multiple calls work correctly")
    return True


def test_softmax_invalid_dim():
    """Test 11: Error handling"""
    print("Testing error handling...")
    
    try:
        softmax = Softmax(dim=5)
        x = Tensor(np.random.randn(2, 3))
        out = softmax(x)
        print("  ✗ Should have failed with invalid dimension")
        return False
    except Exception as e:
        print(f"  ✓ Invalid dimension caught: {type(e).__name__}")
        return True


def main():
    tests = [
        ('Forward Pass Basic', test_softmax_forward_basic),
        ('Forward Pass Different Dim', test_softmax_forward_different_dim),
        ('Forward Pass 1D', test_softmax_forward_1d),
        ('Forward Pass 3D', test_softmax_forward_3d),
        ('Numerical Stability', test_softmax_numerical_stability),
        ('Backward Pass Basic', test_softmax_backward_basic),
        ('Backward with Loss', test_softmax_backward_with_loss),
        ('Softmax Properties', test_softmax_properties),
        ('Computation Graph', test_softmax_computation_graph),
        ('Multiple Calls', test_softmax_multiple_calls),
        ('Invalid Dimension', test_softmax_invalid_dim),
    ]
    
    passed = 0
    failed = 0
    failed_tests = []
    
    print("\n" + "=" * 60)
    print("SOFTMAX ACTIVATION COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    for test_name, test_func in tests:
        success, _ = run_test(test_name, test_func)
        if success:
            passed += 1
        else:
            failed += 1
            failed_tests.append(test_name)
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed_tests:
        print("\nFailed tests:")
        for test in failed_tests:
            print(f"  ✗ {test}")
    
    print("\n" + "=" * 60)
    
    if failed == 0:
        print("\n🎉 ALL SOFTMAX TESTS PASSED! 🎉")
    else:
        print(f"\n❌ {failed} Test(s) Failed")
    
    print("=" * 60)
    input("\nPress Enter to exit...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
        input("\nPress Enter to exit...")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        traceback.print_exc()
        input("\nPress Enter to exit...")