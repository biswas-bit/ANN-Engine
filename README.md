# ANN Engine – A Minimal Neural Network Training Framework from Scratch

## OverView
ANN Engine is a lightweight neural network training framework built from scratch using NumPy.

The objective of this project is to understand and implement the core mechanics of deep learning frameworks such as:
- Forward propagation
- Backpropagation
- Automatic gradient computation
- Optimization algorithms
- Modular layer design
- Training loop abstraction

This project focuses on clarity, modularity, and mathematical correctness rather than performance.

## Motivation
Modern deep learning frameworks abstract away gradient computation and optimization.

To deeply understand how neural networks train internally, this project re-implements:

- Linear layers
- Activation functions
- Loss functions
- Optimizers
- Backpropagation logic
- Model class with fit/predict interface

The goal is to bridge the gap between theory and production frameworks.

## Folder Structure

```python 
ann-engine/
│
├── README.md
├── pyproject.toml        
├── requirements.txt
│
├── ann_engine/             # Core library 
│   ├── __init__.py
│
│   ├── core/
│   │   ├── __init__.py
│   │   ├── parameter.py    # Trainable weights & gradients
│   │   └── tensor.py       
│
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── base.py         # Layer abstract class
│   │   ├── dense.py
│   │   └── activations.py  # ReLU, Sigmoid, Softmax
│
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── mse.py
│   │   └── cross_entropy.py
│
│   ├── optimizers/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── sgd.py
│   │   └── adam.py
│
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── model.py        # Sequential-like container
│   │   └── trainer.py      # Training loop
│
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── initializers.py
│   │   ├── metrics.py
│   │   └── checks.py       # Gradient checking (later)
│
│   └── exceptions.py       # Custom framework errors
│
├── examples/
│   ├── binary_classification.py
│   ├── multiclass_classification.py
│   └── regression.py
│
├── tests/                 
    ├── test_dense.py
    ├── test_losses.py
    └── test_optimizers.py
