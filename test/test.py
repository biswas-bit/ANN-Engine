from ann_engine.core.parameter import Parameter
import numpy as np

W = Parameter(np.array([[1., 2.],
                        [3., 4.]]))
print("Data:\n", W.data)
print("Grad:\n", W.grad)
print("Required Grad:", W.required_grad)

x = Parameter(np.array([[1., 1.],
                        [0., 1.]]))
z= W + x
print("\nResult of w+x (Tensor):",z)
print("z is Parameter? :", isinstance(z, Parameter))
print("\nz.data:\n", z.data)

z.backward()
print("\ngrad after backward:")
print("\nGrad of W:\n", W.grad)
print("Grad of x:\n", x.grad)   
