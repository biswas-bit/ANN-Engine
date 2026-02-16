from ann_engine.core.tensor import Tensor

a = Tensor(2.0)
b = Tensor(3.0)
c = Tensor(4.0)
d = a*b+c*b+a

print(d.grad)