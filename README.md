# ANN Engine – A Minimal Neural Network Training Framework from Scratch

<p align="center">
 <img src="ann_engine.png" alt="Missing Value Count Plot" width="600">
 </p>

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

```
ann_engine/
├── __init__.py
│
├── core/
│   ├── __init__.py
│   ├── parameter.py
│   └── tensor.py
│
├── datasets/
│   ├── __init__.py
│   ├── base.py
│   └── datasets.py
│
├── engine/
│   ├── __init__.py
│   └── models.py
│
├── layers/
│   ├── __init__.py
│   ├── activations.py
│   ├── base.py
│   ├── dense.py
│   ├── layers.py
│   └── sequential.py
│
├── losses/
│   ├── __init__.py
│   ├── base.py
│   └── loss.py
│
├── optimizers/
│   ├── __init__.py
│   ├── adagrad.py
│   ├── adam.py
│   ├── base.py
│   ├── nag.py
│   ├── rsmprop.py
│   └── sgd.py
│
└── utils/
    └── __init__.py
```