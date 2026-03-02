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

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Architecture Overview](#architecture-overview)
4. [API Reference](#api-reference)
   - [Models](#models)
   - [Layers](#layers)
   - [Loss Functions](#loss-functions)
   - [Optimizers](#optimizers)
   - [Datasets](#datasets)
5. [Examples](#examples)
   - [Classification Example](#classification-example)
   - [Regression Example](#regression-example)
   - [Custom Model Example](#custom-model-example)
6. [Folder Structure](#folder-structure)

---

## Installation

### Prerequisites

- Python 3.8+
- NumPy

### Install from Source

```
bash
# Clone the repository
git clone https://github.com/yourusername/ann_engine.git
cd ann_engine

# Install in development mode
pip install -e .
```

### Using pip (if published)

```
bash
pip install ann-engine
```

---

## Quick Start

Here's a minimal example to get you started with a neural network for digit classification:

```
python
import numpy as np
from ann_engine.layers import Sequential, Dense
from ann_engine.optimizers import Adam
from ann_engine.losses import CrossEntropyLoss
from ann_engine.datasets import Datasets

# 1. Generate sample data (handwritten digits - 8x8 pixels)
X, y = Datasets.make_spirals(n=1000, n_classes=10, noise=0.2)

# One-hot encode labels
from ann_engine.core import Tensor
n_classes = len(np.unique(y))
y_onehot = np.zeros((len(y), n_classes))
y_onehot[np.arange(len(y)), y] = 1

# 2. Build the model
model = Sequential([
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(n_classes, activation='softmax'),
])

# 3. Compile the model
model.compile(
    optimizer=Adam(lr=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Train the model
history = model.fit(X, y_onehot, epochs=50, batch_size=32, verbose=1)

# 5. Make predictions
predictions = model.predict(X)
predicted_classes = np.argmax(predictions.data, axis=1)
accuracy = np.mean(predicted_classes == y)
print(f"Training Accuracy: {accuracy:.4f}")
```

---

## Architecture Overview

ANN Engine follows a modular architecture similar to PyTorch/Keras:

```
┌─────────────────────────────────────────────────────────────┐
│                        Model (Sequential)                   │
├─────────────────────────────────────────────────────────────┤
│  Layer 1      Layer 2      Layer 3      ...    Output      │
│  (Dense)      (Dense)      (Dense)               (Loss)    │
├─────────────────────────────────────────────────────────────┤
│                    Forward Propagation                      │
│                    (x → layer → ... → y)                    │
├─────────────────────────────────────────────────────────────┤
│                    Backward Propagation                     │
│                    (dy ← layer ← ... ← dx)                 │
├─────────────────────────────────────────────────────────────┤
│                      Optimizer (Adam, SGD, etc.)           │
│                  (updates parameters using gradients)       │
└─────────────────────────────────────────────────────────────┘
```

### Core Components:

1. **Tensor**: Custom automatic differentiation tensor class
2. **Layers**: Dense layers with various activation functions
3. **Loss**: Various loss functions for training
4. **Optimizers**: Different optimization algorithms
5. **Datasets**: Built-in dataset generators for testing

---

## API Reference

### Models

#### Sequential

A linear stack of layers with a Keras-like interface.

```
python
from ann_engine.layers import Sequential, Dense
from ann_engine.optimizers import Adam
from ann_engine.losses import CrossEntropyLoss

model = Sequential([
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax'),
])
```

**Methods:**

| Method | Description |
|--------|-------------|
| `add(layer)` | Add a layer to the model |
| `compile(optimizer, loss, metrics)` | Configure the model for training |
| `fit(x, y, batch_size, epochs, validation_split, verbose)` | Train the model |
| `predict(x)` | Generate predictions |
| `evaluate(x, y, batch_size)` | Evaluate the model |
| `summary()` | Print model architecture |

---

### Layers

#### Dense Layer

Fully connected layer with optional activation.

```
python
from ann_engine.layers import Dense

# Basic usage
layer = Dense(128, activation='relu')

# With custom initialization
layer = Dense(256, activation='sigmoid', use_bias=True)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `units` | int | Number of neurons |
| `activation` | str or None | Activation function ('relu', 'sigmoid', 'tanh', 'softmax', etc.) |
| `use_bias` | bool | Whether to include bias (default: True) |

#### Activation Functions

Available activation functions:

| Function | Description | Use Case |
|----------|-------------|----------|
| `ReLU(x)` | Rectified Linear Unit | Default for hidden layers |
| `Sigmoid(x)` | Sigmoid activation | Binary classification output |
| `Tanh(x)` | Hyperbolic Tangent | Hidden layers (less common) |
| `LeakyReLU(x)` | Leaky ReLU | Prevents dying ReLU problem |
| `ELU(x)` | Exponential Linear Unit | Smooth ReLU alternative |
| `Softmax(x)` | Softmax activation | Multi-class classification |
| `LogSoftmax(x)` | Log Softmax | Numerical stability |
| `Softplus(x)` | Softplus activation | Smooth ReLU alternative |
| `Swish(x)` | Swish activation | Self-gated activation |
| `GELU(x)` | Gaussian Error Linear Unit | BERT-style models |
| `Identity(x)` | No activation | Regression output layer |

**Usage:**

```
python
from ann_engine.layers import Dense, ReLU, Sigmoid, Tanh

model = Sequential([
    Dense(128),
    ReLU(),           # Can use as separate layer
    Dense(64),
    Sigmoid(),        # Or use activation in Dense
])
```

---

### Loss Functions

Available loss functions:

| Loss Function | Description | Use Case |
|---------------|-------------|----------|
| `MSELoss` | Mean Squared Error | Regression |
| `CrossEntropyLoss` | Categorical Cross-Entropy | Multi-class classification |
| `NLLLoss` | Negative Log Likelihood | Classification with log probabilities |
| `BCELoss` | Binary Cross-Entropy | Binary classification |
| `BCEWithLogitsLoss` | BCE with sigmoid baked in | Binary classification |
| `HuberLoss` | Huber loss (L1 + L2) | Regression, outlier robust |

**Usage:**

```
python
from ann_engine.losses import MSELoss, CrossEntropyLoss, BCELoss

# Via string (recommended)
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy')

# Via class
model.compile(optimizer=Adam(lr=0.001), loss=CrossEntropyLoss())

# Available string aliases:
# 'mse', 'mean_squared_error' → MSELoss
# 'categorical_crossentropy', 'cross_entropy' → CrossEntropyLoss
# 'binary_crossentropy', 'bce' → BCELoss
# 'bce_with_logits', 'bce_logits' → BCEWithLogitsLoss
# 'huber' → HuberLoss
```

---

### Optimizers

Available optimizers:

| Optimizer | Description |
|-----------|-------------|
| `SGD` | Stochastic Gradient Descent |
| `SGDWithMomentum` | SGD with momentum |
| `NAG` | Nesterov Accelerated Gradient |
| `AdaGrad` | Adaptive Gradient |
| `Adam` | Adaptive Moment Estimation |
| `RMSProp` | Root Mean Square Propagation |

**Usage:**

```
python
from ann_engine.optimizers import SGD, Adam, AdaGrad, RMSProp

# Adam (default choice)
optimizer = Adam(lr=0.001)

# SGD with momentum
optimizer = SGD(lr=0.01, momentum=0.9)

# RMSProp
optimizer = RMSProp(lr=0.001, decay=0.9)

# AdaGrad
optimizer = AdaGrad(lr=0.01)

# NAG (Nesterov Accelerated Gradient)
optimizer = NAG(lr=0.01, momentum=0.9)
```

**Optimizer Parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `lr` | Learning rate | Required |
| `momentum` | Momentum factor (SGD, NAG) | 0.0 |
| `decay` | Learning rate decay (RMSProp, Adam) | 0.0 |
| `beta1` | First moment decay (Adam) | 0.9 |
| `beta2` | Second moment decay (Adam) | 0.999 |
| `epsilon` | Small constant for numerical stability | 1e-8 |

---

### Datasets

Built-in dataset generators for testing:

```
python
from ann_engine.datasets import Datasets
```

| Method | Description |
|--------|-------------|
| `make_spirals(n, n_classes, noise)` | Generate spiral dataset |
| `make_moons(n, noise)` | Generate moons dataset |
| `make_circles(n, noise, factor)` | Generate circles dataset |
| `make_xor(n, noise)` | Generate XOR dataset |
| `make_regression(n, noise)` | Generate regression dataset |
| `make_imbalanced(n, imbalance)` | Generate imbalanced dataset |
| `make_high_dimensional(n)` | Generate high-dimensional dataset |

---

## Examples

### Classification Example

Complete example with MNIST-style data:

```
python
import numpy as np
from ann_engine.layers import Sequential, Dense
from ann_engine.optimizers import Adam
from ann_engine.losses import CrossEntropyLoss
from ann_engine.datasets import Datasets
from ann_engine.core import Tensor

# Generate sample data (replace with real data for production)
X, y = Datasets.make_spirals(n=2000, n_classes=10, noise=0.2)

# One-hot encode labels
n_classes = len(np.unique(y))
y_onehot = np.zeros((len(y), n_classes))
y_onehot[np.arange(len(y)), y] = 1

# Build model
model = Sequential([
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(n_classes, activation='softmax'),
])

# Compile
model.compile(
    optimizer=Adam(lr=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary
model.summary()

# Train
history = model.fit(
    X, y_onehot, 
    epochs=100, 
    batch_size=32, 
    validation_split=0.2,
    verbose=1
)

# Evaluate
results = model.evaluate(X, y_onehot)
print(f"Final Loss: {results[0]:.4f}")
print(f"Accuracy: {results[1]:.4f}")
```

### Regression Example

Predicting continuous values:

```
python
import numpy as np
import matplotlib.pyplot as plt
from ann_engine.layers import Sequential, Dense
from ann_engine.optimizers import Adam
from ann_engine.losses import MSELoss
from ann_engine.datasets import Datasets

# Generate regression data
X, y = Datasets.make_regression(n=1000, noise=0.2)

# Build model
model = Sequential([
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='identity'),  # Linear output for regression
])

# Compile
model.compile(
    optimizer=Adam(lr=0.01),
    loss='mse',
    metrics=['mae']
)

# Train
history = model.fit(X, y, epochs=100, batch_size=32, verbose=1)

# Predict
predictions = model.predict(X)

# Plot results
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(X.flatten()[:100], y.flatten()[:100], alpha=0.5, label='Actual')
plt.scatter(X.flatten()[:100], predictions.data.flatten()[:100], alpha=0.5, label='Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Regression Results')
plt.legend()
plt.show()
```

### Custom Model Example

For more complex architectures, use the `Model` class:

```
python
from ann_engine.layers import Dense, ReLU, Sigmoid
from ann_engine.engine import Model as BaseModel

class CustomNetwork(BaseModel):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = Dense(hidden_dim, activation='relu')
        self.fc2 = Dense(hidden_dim, activation='relu')
        self.fc3 = Dense(output_dim, activation='sigmoid')
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# Usage
model = CustomNetwork(input_dim=784, hidden_dim=256, output_dim=1)
model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy')
history = model.fit(X_train, y_train, epochs=10, batch_size=32)
```

---

## Folder Structure

```
ann_engine/
├── __init__.py
│
├── core/
│   ├── __init__.py
│   ├── parameter.py      # Parameter wrapper
│   └── tensor.py         # Tensor with autograd
│
├── datasets/
│   ├── __init__.py
│   ├── base.py           # Dataset generators
│   └── datasets.py      # Dataset class
│
├── engine/
│   ├── __init__.py
│   └── models.py         # Model base class
│
├── layers/
│   ├── __init__.py
│   ├── activations.py    # Activation functions
│   ├── base.py           # Layer base class
│   ├── dense.py          # Dense layer
│   ├── layers.py         # Layer exports
│   └── sequential.py     # Sequential model
│
├── losses/
│   ├── __init__.py
│   ├── base.py           # Loss base class
│   └── loss.py           # Loss functions
│
├── optimizers/
│   ├── __init__.py
│   ├── adagrad.py        # AdaGrad optimizer
│   ├── adam.py           # Adam optimizer
│   ├── base.py           # Optimizer base class
│   ├── nag.py            # NAG optimizer
│   ├── rsmprop.py        # RMSProp optimizer
│   └── sgd.py            # SGD optimizer
│
└── utils/
    └── __init__.py
```

---

## Performance Tips

1. **Learning Rate**: Start with `lr=0.001` for Adam, `lr=0.01` for SGD
2. **Batch Size**: Use 32-64 for small datasets, up to 256 for large ones
3. **Architecture**: Start simple (1-2 hidden layers), increase complexity as needed
4. **Validation**: Always use `validation_split` to monitor overfitting
5. **Early Stopping**: Monitor validation loss and stop if it increases

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## License

MIT License

---

## Credits

Built with ❤️ by [Biswas pokhrel](https://github.com/yourusername)

This framework is for educational purposes to understand the internals of deep learning frameworks.
