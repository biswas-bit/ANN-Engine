# test_adagrad.py (simplified without retain_graph)
import numpy as np
from ann_engine.core.parameter import Parameter
from ann_engine.optimizers.adagrad import AdaGrad

def test_simple_quadratic():
    """Test AdaGrad on simple quadratic: (w - 5)^2"""
    print("=" * 50)
    print("TEST 1: Simple Quadratic (w - 5)^2")
    print("=" * 50)
    
    w = Parameter(np.array([0.0]))
    optimizer = AdaGrad([w], lr=0.1)
    
    print(f"Initial w: {w.data}")
    
    for step in range(10):
        optimizer.zero_grad()
        loss = (w - 5) ** 2
        loss.backward()
        
        print(f"\nStep {step}:")
        print(f"  Before update: w={w.data[0]:.6f}, grad={w.grad[0]:.6f}")
        print(f"  Accumulated G: {optimizer.G[0][0]:.6f}")
        
        optimizer.step()
        
        print(f"  After update:  w={w.data[0]:.6f}")
    
    print(f"\nFinal w: {w.data[0]:.6f} (expected ~5.0)")
    return w.data[0]

def test_sparse_features():
    """Test AdaGrad on sparse features - simplified version"""
    print("\n" + "=" * 50)
    print("TEST 2: Sparse Features Test")
    print("=" * 50)
    
    # Simulate sparse features - some parameters get frequent updates, some rare
    w_frequent = Parameter(np.array([0.0]))  # Frequently updated
    w_rare = Parameter(np.array([0.0]))      # Rarely updated
    
    optimizer = AdaGrad([w_frequent, w_rare], lr=0.1)
    
    print("AdaGrad should handle different update frequencies well")
    
    for step in range(20):
        # Zero gradients for all parameters
        optimizer.zero_grad()
        
        # Compute loss for frequent parameter (always updated)
        loss_frequent = (w_frequent - 5) ** 2
        loss_frequent.backward()
        
        # Store frequent gradient
        grad_frequent = w_frequent.grad.copy() if w_frequent.grad is not None else None
        
        # Zero gradients for rare parameter separately
        w_rare.grad = np.zeros_like(w_rare.data)
        
        # Compute loss for rare parameter only every 5 steps
        if step % 5 == 0:
            loss_rare = (w_rare - 5) ** 2
            loss_rare.backward()
            print(f"\nStep {step} (rare param updated):")
        else:
            print(f"\nStep {step} (rare param NOT updated):")
        
        print(f"  w_frequent: {w_frequent.data[0]:.6f} (grad={grad_frequent[0] if grad_frequent is not None else 0:.6f})")
        print(f"  w_rare: {w_rare.data[0]:.6f} (grad={w_rare.grad[0]:.6f})")
        print(f"  G_frequent: {optimizer.G[0][0]:.6f}, G_rare: {optimizer.G[1][0]:.6f}")
        
        optimizer.step()
    
    return w_frequent.data[0], w_rare.data[0]

def test_2d_parameter():
    """Test AdaGrad on 2D parameter"""
    print("\n" + "=" * 50)
    print("TEST 3: 2D Parameter")
    print("=" * 50)
    
    w = Parameter(np.array([[1.0, 2.0], [3.0, 4.0]]))
    target = np.array([[5.0, 5.0], [5.0, 5.0]])
    
    optimizer = AdaGrad([w], lr=0.1)
    
    print("Initial w:")
    print(w.data)
    print("Target:")
    print(target)
    
    for step in range(20):
        optimizer.zero_grad()
        loss = ((w - target) ** 2).sum()
        loss.backward()
        
        if step % 5 == 0:
            print(f"\nStep {step}: Loss={loss.data:.6f}")
            print(f"  Accumulated G:")
            print(optimizer.G[0])
        
        optimizer.step()
    
    print("\nFinal w:")
    print(w.data)
    print("\nError from target:")
    print(w.data - target)
    
    return w.data

def test_compare_sgd_adagrad():
    """Compare AdaGrad with SGD on ill-conditioned problem"""
    print("\n" + "=" * 50)
    print("TEST 4: AdaGrad vs SGD on Ill-conditioned Problem")
    print("=" * 50)
    
    from ann_engine.optimizers.sgd import SGD
    
    # Ill-conditioned problem: different scales
    w1_adagrad = Parameter(np.array([0.0]))
    w2_adagrad = Parameter(np.array([0.0]))
    w1_sgd = Parameter(np.array([0.0]))
    w2_sgd = Parameter(np.array([0.0]))
    
    optimizer_adagrad = AdaGrad([w1_adagrad, w2_adagrad], lr=0.1)
    optimizer_sgd = SGD([w1_sgd, w2_sgd], lr=0.01)
    
    print("Problem: w1 should go to 5 (small scale), w2 to 0.1 (large scale difference)")
    
    for step in range(50):
        # AdaGrad updates
        optimizer_adagrad.zero_grad()
        loss_adagrad = (w1_adagrad - 5) ** 2 + 100 * (w2_adagrad - 0.1) ** 2
        loss_adagrad.backward()
        optimizer_adagrad.step()
        
        # SGD updates
        optimizer_sgd.zero_grad()
        loss_sgd = (w1_sgd - 5) ** 2 + 100 * (w2_sgd - 0.1) ** 2
        loss_sgd.backward()
        optimizer_sgd.step()
        
        if step % 10 == 0:
            print(f"\nStep {step}:")
            print(f"  AdaGrad: w1={w1_adagrad.data[0]:.6f}, w2={w2_adagrad.data[0]:.6f}")
            print(f"  SGD:     w1={w1_sgd.data[0]:.6f}, w2={w2_sgd.data[0]:.6f}")
    
    print(f"\nFinal AdaGrad: w1={w1_adagrad.data[0]:.6f}, w2={w2_adagrad.data[0]:.6f}")
    print(f"Final SGD:     w1={w1_sgd.data[0]:.6f}, w2={w2_sgd.data[0]:.6f}")
    
    return (w1_adagrad.data[0], w2_adagrad.data[0]), (w1_sgd.data[0], w2_sgd.data[0])

if __name__ == "__main__":
    print("ADAGRAD OPTIMIZER TEST SUITE")
    print("=" * 50)
    
    try:
        test_simple_quadratic()
        test_sparse_features()
        test_2d_parameter()
        test_compare_sgd_adagrad()
        
        print("\n" + "=" * 50)
        print("✓ ALL TESTS PASSED!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to exit...")