from ann_engine.losses.loss import CrossEntropyLoss 
from ann_engine.core import Tensor

y_true = Tensor([1.0, 2.0, 3.0])
y_pred = Tensor([1.5, 2.5, 3.5])

try :
  loss_fn = CrossEntropyLoss(reduction='mean')
  loss = loss_fn(y_pred, y_true)
  print("Cross Entropy Loss:", loss.data)

  loss.backward()
  print("Gradient w.r.t y_pred:", y_pred.grad.data)
except Exception as e:
  print("Error during loss computation or backpropagation:", e)
input("press enter to continue..")
