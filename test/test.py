# test_adam.py
import numpy as np
from ann_engine.core.parameter import Parameter
from ann_engine.optimizers.adam import Adam

def test_simple_quadratic():
    """Test Adam on simple quadratic: (w - 5)^2"""
    print("=" * 50)
    print("TEST 1: Simple Quadratic (w - 5)^2")
    print("=" * 50)
    
    w = Parameter(np.array([0.0]))
    # Use 'lr' instead of 'learning_rate'
    optimizer = Adam([w], lr=0.1)
    
    print(f"Initial w: {w.data}")
    
    for step in range(10):
        optimizer.zero_grad()
        loss = (w - 5) ** 2
        loss.backward()
        
        print(f"\nStep {step}:")
        print(f"  Before update: w={w.data[0]:.6f}, grad={w.grad[0]:.6f}")
        
        optimizer.step()
        
        print(f"  After update:  w={w.data[0]:.6f}")
        print(f"  Target: 5.0, Error: {abs(w.data[0] - 5.0):.6f}")
    
    print(f"\nFinal w: {w.data[0]:.6f} (expected ~5.0)")
    return w.data[0]

def test_multi_parameter():
    """Test Adam on multiple parameters with different scales"""
    print("\n" + "=" * 50)
    print("TEST 2: Multiple Parameters")
    print("=" * 50)
    
    w1 = Parameter(np.array([2.0]))
    w2 = Parameter(np.array([-3.0]))
    w3 = Parameter(np.array([10.0]))
    
    optimizer = Adam([w1, w2, w3], lr=0.01)
    
    print("Initial values:")
    print(f"  w1: {w1.data[0]}, w2: {w2.data[0]}, w3: {w3.data[0]}")
    
    for step in range(20):
        optimizer.zero_grad()
        
        # Different losses for each parameter
        loss1 = (w1 - 5) ** 2  # Target: 5
        loss2 = (w2 + 2) ** 2   # Target: -2
        loss3 = (w3 - 0) ** 2   # Target: 0
        
        loss = loss1 + loss2 + loss3
        loss.backward()
        
        if step % 5 == 0:
            print(f"\nStep {step}:")
            print(f"  w1={w1.data[0]:.6f} (grad={w1.grad[0]:.6f})")
            print(f"  w2={w2.data[0]:.6f} (grad={w2.grad[0]:.6f})")
            print(f"  w3={w3.data[0]:.6f} (grad={w3.grad[0]:.6f})")
        
        optimizer.step()
    
    print(f"\nFinal values:")
    print(f"  w1: {w1.data[0]:.6f} (target: 5.0)")
    print(f"  w2: {w2.data[0]:.6f} (target: -2.0)")
    print(f"  w3: {w3.data[0]:.6f} (target: 0.0)")
    
    return w1.data[0], w2.data[0], w3.data[0]

if __name__ == "__main__":
    print("ADAM OPTIMIZER TEST SUITE")
    print("=" * 50)
    
    try:
        # Run tests
        test_simple_quadratic()
        test_multi_parameter()
        
        print("\n" + "=" * 50)
        print("✓ ALL TESTS PASSED!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to exit...")