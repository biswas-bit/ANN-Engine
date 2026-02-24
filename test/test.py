import numpy as np
import sys
import traceback

def pause_on_error():
    print("\n" + "=" * 60 )
    input("Press Enter to continue after error...")
    

try:
    from ann_engine.core import Tensor, Parameter
    from ann_engine.layers import ReLU
except ImportError as e:
    print(f"Falied to Import lbraries : {e}")
    pause_on_error()
    sys.exit(1)
    
def run_test(test_name, test_func):
    print(f"\n" + "=" * 50)
    print(f"Running test : {test_name}")
    print(f"\n" + "=" * 50)
    
    try:
        result = test_func()
        print(f"\n Test Passed : {test_name}")
        return True, result
    except AssertionError as e:
        print(f"\n test failed (Assertion Error): {test_name}")
        print(f" {e}")
        print("\n" + "-" * 50)
        traceback.print_exc()
        pause_on_error()
        return False , None
    
    
    except Exception as e:
        print('\n Test  Failed (Exception): {test_name}')
        print(f" {e}")
        print("\n" + "-" * 50)
        traceback.print_exc()
        pause_on_error()
        return False , None
    

def test_relu_forward_basic():
    """ Test 1: Basic Relu forward pass """
    print("Testing basic ReLU forward pass")
    
    relu = ReLU()
    print(f"ReLU instance: {relu}")
    
    # Test with positive, negative, and Zero
    x_data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    x = Tensor(x_data)
    out = relu(x)
    
    # expected result
    expected = np.array([0.0, 0.0,0.0, 1.0, 2.0])
    
    print(f"Input: {x_data}")
    print(f"output: {out.data}")
    print(f" Expected : {expected}")
    
    assert np.allclose(out.data, expected), f"Expected {expected}, got {out.data}"
    assert out._op == "ReLU", f"Expected op 'ReLU', got {out._op}"
    assert x in out._prev, "Input should be computation graph"
    print("forward pass correct")
    print("computation graph build correct")
    return True

def relu_test_forward_2d():
    print("Testing  forward pass with 2d input...")
    relu = ReLU()
    
    x_data = np.array([
        [-1.0, -0.5, 0.0, 0.5],
        [1.0, 1.5, -2.0, -1.0],
        [0.0, -3.0, 4.0, 5.0]
    ])
    
    x = Tensor(x_data)
    out = relu(x)
    expected = np.maximum(0, x_data)
    print(f" Input shape: {x_data.shape}")
    print(f" Output shape: {out.data.shape}")
    print(f" sample output:n\n{out.data}")
    
    assert out.data.shape == x_data.shape, f"Shape mismatch: {out.data.shape} vs {x_data.shape}"
    assert np.allclose(out.data, expected), "OutPut doesn't match expected"
    print(" 2d forward pass correct")
    return True

def test_relu_backward_basic():
    print("Testing ReLU backward pass with basic value...")
    relu = ReLU()
    x = Tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    out = relu(x)
    out.grad = np.ones_like(out.data)
    out._backward()
    expected_grad = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    print(f"  Input: {x.data}")
    print(f"  Output: {out.data}")
    print(f"  Computed gradients: {x.grad}")
    print(f"  Expected gradients: {expected_grad}")
    
    assert np.allclose(x.grad, expected_grad), f"Expected {expected_grad}, got {x.grad}"
    print("  ✓ Backward pass correct")
    
    return True

def test_relu_backward_with_loss():
    print("Testing ReLU backward pass with loss.backward()...")
    relu = ReLU()
    x = Parameter(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    out = relu(x)
    loss = out.sum()
    loss.backward()
    expected_grad = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    print(f"  Input: {x.data}")
    print(f"  Output: {out.data}")
    print(f"  Loss: {loss.data}")
    print(f"  Computed gradients: {x.grad}")
    print(f"  Expected gradients: {expected_grad}")
    
    assert np.allclose(x.grad, expected_grad), f"Expected {expected_grad}, got {x.grad}"
    print("  ✓ Backward pass with loss.backward() correct")
    return True

def test_relu_backward_2d():
    print("Testing ReLU backward pass with 2D input...")
    relu = ReLU()
    x_data = np.array([
        [-1.0, -0.5, 0.0, 0.5],
        [1.0, 1.5, -2.0, -1.0],
        [0.0, -3.0, 4.0, 5.0]
    ])
    x = Parameter(x_data)
    out = relu(x)
    loss = out.sum()
    loss.backward()
    expected_grad = (x_data > 0).astype(np.float32)
    print(f"  Input shape: {x_data.shape}")
    print(f"  Output shape: {out.data.shape}")
    print(f"  Computed gradients shape: {x.grad.shape}")
    print(f"  Sample gradients:\n{x.grad}")
    assert x.grad.shape == x_data.shape, f"Gradient shape mismatch: {x.grad.shape} vs {x_data.shape}"
    assert np.allclose(x.grad, expected_grad), "Gradients don't match expected"
    print("2D backward pass correct")
    
    return True

def test_relu_computation_graph():
    print("Testing ReLU computation graph...")
    
    relu = ReLU()
    
    x = Tensor(np.array([-1.0, 0.0, 1.0]))
    y = Tensor(np.array([2.0, 2.0, 2.0]))
    
    out = relu(x) + y
    print(f"  Input x: {x.data}")
    print(f"  Input y: {y.data}")
    print(f"  Output: {out.data}")
    print(f"  Output operation: {out._op}")
    print(f"  Output parents: {[id(p) for p in out._prev]}")
    
    assert out._op == '+', "Output should be from addition"
    
    # Find the ReLU node in the graph
    relu_node = None
    for child in out._prev:
        if child._op == 'ReLU':
            relu_node = child
            break
    
    assert relu_node is not None, "ReLU node not found in computation graph"
    assert x in relu_node._prev, "Input x should be parent of ReLU node"
    print("Computation graph built correctly")
    
    return True
    
    
def main():
    tests = [
        ('Forward Pass Basic',test_relu_forward_basic),
        ('Forward pass Batch', relu_test_forward_2d),
        ('Backward pass for basic',test_relu_backward_basic),
        ('Backward with loss',  test_relu_backward_with_loss),
        ('Backward test with 2d', test_relu_backward_2d),
        ('Testing Computation graph',test_relu_computation_graph),
    ]
    
    passed =0
    failed =0
    failed_tests = []
    for test_name, test_func in tests:
        success, _ = run_test(test_name, test_func)
        if success:
            passed +=1
        else:
            failed +=1
            failed_tests.append(test_name)
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY - RELU")
    print("=" * 50)
    print(f"Total Test : {len(tests)}")
    print(f"passed: {passed}")
    print(f"failed : {failed}")
    
    if failed_tests:
        print("\n Failed tests:")
        for test in failed_tests:
            print(f"=>{test}")
    
    print("\n" + "=" * 50)
    
    if failed == 0:
        print("\n ALL RELU TESTS PASSED")
        print("ReLU Implementation is working correctly")
    
    else:
        print(f"\n => {failed} Tests Failed. Check the erros above.")
        print("The Programme paused after each error so you can see what happened")
    
    print("\n" + "=" * 50)
    input("\nPress Enter to exit...") 
    
    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
        input("\nPress Enter to exit...")
    
    except Exception as e:
        print(f"\n Unexpected error in main : {e}")
        traceback.print_exc()
        input("\npress enter to exit")