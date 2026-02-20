import numpy as np
from ann_engine.core.parameter import Parameter
from ann_engine.optimizers.sgd import SGD, SGDWithMomentum


W = Parameter(np.array([[2.0, 3.0],
                        [4.0, 5.0]]))

loss = (W * 2).sum()


loss.backward()

print("Before step:")
print("W.data:\n", W.data)
print("W.grad:\n", W.grad)


optimizer = SGD([W], lr=0.1)
optimizer2 = SGDWithMomentum([W], lr=0.1, momentum=0.9)

optimizer.step()
optimizer2.step()

print("\nAfter step:")
print("W.data:\n", W.data)

input()