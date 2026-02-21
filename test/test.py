from ann_engine.core.parameter import Parameter
from ann_engine.optimizers.adagrad import AdaGrad
import numpy as np

# Parameter
w = Parameter(np.array([0.0]))

# Optimizer
optimizer = AdaGrad([w], lr=0.1)

# Training loop
for step in range(5):
    optimizer.zero_grad()

    # Forward
    loss = (w - 5) ** 2
    loss.backward()

    # Print before step
    print(f"Step {step} before update: w={w.data}, grad={w.grad}")

    # Update
    optimizer.step()

    # Print after step
    print(f"Step {step} after update: w={w.data}")
input()