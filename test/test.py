from ann_engine.losses.loss import MSELoss 
from ann_engine.core import Tensor, Parameter

# Regression data
y_true = Tensor([1.0, 2.0, 3.0])
y_pred = Parameter([1.5, 2.5, 3.5]) 

try:
    loss_fn = MSELoss(reduction='mean')
    loss = loss_fn(y_pred, y_true)
    print("MSE Loss:", loss.data)

    loss.backward()
    print("Gradient w.r.t y_pred:", y_pred.grad)
except Exception as e:
    print("Error during loss computation or backpropagation:", e)
    import traceback
    traceback.print_exc()

input("Press Enter to continue...")