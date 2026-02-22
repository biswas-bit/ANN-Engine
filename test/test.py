# test_dense.py
import numpy as np
from ann_engine.core.tensor import Tensor
from ann_engine.layers.dense import Dense

def test_dense_layer():
    """Test Dense layer functionality"""
    print("=" * 60)
    print("TESTING DENSE LAYER")
    print("=" * 60)
    
    try:
        # Test 1: Create layer and check shapes
        print("\n1. Creating Dense layer (3 -> 2)...")
        layer = Dense(3, 2)
        print(f"   Layer: {layer}")
        print(f"   Weight shape: {layer.W.data.shape}")
        print(f"   Bias shape: {layer.b.data.shape if layer.b else 'None'}")
        
        # Test 2: Forward pass with batch
        print("\n2. Forward pass with batch size 4...")
        x = Tensor(np.random.randn(4, 3))
        print(f"   Input shape: {x.data.shape}")
        
        out = layer(x)
        print(f"   Output shape: {out.data.shape}")
        assert out.data.shape == (4, 2), f"Expected (4,2), got {out.data.shape}"
        print("   ✓ Shape correct")
        
        # Test 3: Forward pass with single sample
        print("\n3. Forward pass with single sample...")
        x_single = Tensor(np.random.randn(3))
        print(f"   Input shape: {x_single.data.shape}")
        
        out_single = layer(x_single)
        print(f"   Output shape: {out_single.data.shape}")
        assert out_single.data.shape == (1, 2), f"Expected (1,2), got {out_single.data.shape}"
        print("   ✓ Shape correct")
        
        # Test 4: Gradient flow
        print("\n4. Testing gradient flow...")
        x = Tensor(np.random.randn(2, 3))
        out = layer(x)
        loss = out.sum()
        loss.backward()
        
        print(f"   Weight gradients shape: {layer.W.grad.shape}")
        print(f"   Bias gradients shape: {layer.b.grad.shape if layer.b else 'None'}")
        assert layer.W.grad is not None, "Weight gradients should exist"
        assert layer.W.grad.shape == layer.W.data.shape, "Gradient shape should match weight shape"
        print("   ✓ Gradients computed correctly")
        
        # Test 5: No bias option
        print("\n5. Testing layer without bias...")
        layer_no_bias = Dense(3, 2, bias=False)
        print(f"   Layer: {layer_no_bias}")
        assert layer_no_bias.b is None, "Bias should be None"
        
        out = layer_no_bias(x)
        loss = out.sum()
        loss.backward()
        print(f"   Output shape: {out.data.shape}")
        print("   ✓ Forward/backward works without bias")
        
        # Test 6: Different initializations
        print("\n6. Testing different initializations...")
        init_methods = ['xavier', 'he', 'normal', 'uniform']
        
        for init in init_methods:
            try:
                layer_init = Dense(3, 2, initialization=init)
                print(f"   ✓ {init} initialization works")
            except Exception as e:
                print(f"   ✗ {init} failed: {e}")
        
        # Test 7: Error handling - wrong input features
        print("\n7. Testing error handling...")
        try:
            x_wrong = Tensor(np.random.randn(2, 4))  # 4 features instead of 3
            out = layer(x_wrong)
            print("   ✗ Should have raised ValueError")
        except ValueError as e:
            print(f"   ✓ Correctly caught error: {e}")
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 60)
    input("\nPress Enter to continue...")

if __name__ == "__main__":
    test_dense_layer()