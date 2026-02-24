# test_relu_simple.py
import numpy as np
from ann_engine.core import Parameter
from ann_engine.layers import ReLU

def test_relu():
    """Simple ReLU test"""
    print("=" * 50)
    print("TESTING RELU ACTIVATION")
    print("=" * 50)
    
    try:
        # Create ReLU
        relu = ReLU()
        print(f"1. Created ReLU: {relu}")
        
        # Test data
        x = Parameter(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
        print(f"\n2. Input: {x.data}")
        
        # Forward pass
        out = relu(x)
        print(f"3. Output: {out.data}")
        print(f"   Expected: [0. 0. 0. 1. 2.]")
        
        # Backward pass
        loss = out.sum()
        loss.backward()
        print(f"\n4. Gradients: {x.grad}")
        print(f"   Expected: [0. 0. 0. 1. 1.]")
        
        # Verify
        assert np.allclose(out.data, [0, 0, 0, 1, 2]), "Forward pass wrong"
        assert np.allclose(x.grad, [0, 0, 0, 1, 1]), "Backward pass wrong"
        
        print("\n✅ ALL TESTS PASSED!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    input("Press Enter to exit...")

if __name__ == "__main__":
    test_relu()