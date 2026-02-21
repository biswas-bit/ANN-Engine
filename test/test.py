from ann_engine.core.parameter import Parameter
from ann_engine.optimizers.nag import NAG
import numpy as np

w = Parameter(np.array([0.0]))
optimizers = NAG([w], lr=0.1, momentum=0.9)

for step in range(5):
    optimizers.zero_grad()
    loss = (w-5)**2
    loss.backward()
    optimizers.step()
    print("grad:",w.grad)
    print(f"step {step}:w={w.data} loss={loss.data}")

input()