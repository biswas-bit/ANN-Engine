# ANN Engine – A Minimal Neural Network Training Framework from Scratch

<p align="center">
 <img src="ann_engine.png" alt="ANN Engine Logo" width="600">
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

## Installation

### From PyPI (Recommended)

```
bash
pip install ann-engine
```

### From Source

```
bash
# Clone the repository
git clone https://github.com/biswas-bit/ANN-Engine.git
cd ANN-Engine

# Install in development mode
pip install -e .
```

### Build from Source

```
bash
# Build the package
python -m build

# Install the built package
pip install dist/ann_engine-0.1.0-py3-none-any.whl
```

## Quick Start

Here's a simple example to get started with the ANN Engine:

```
python
import numpy as np
from ann_engine.layers import Sequential, Dense
from ann_engine.losses import CrossEntropyLoss
from ann_engine.optimizers import Adam

# Create a simple neural network
model = Sequential([
    Dense(128, activation='relu', input_shape=784),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model with loss and optimizer
model.compile(loss='cross_entropy', optimizer=Adam(learning_rate=0.001))

# Train the model
# X_train: training data (n_samples, 784)
# y_train: training labels (n_samples,)
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Make predictions
predictions = model.predict(X_test)
```

## Features

### 🧠 Neural Network Layers

| Layer | Description |
|-------|-------------|
| `Dense` | Fully connected linear layer |
| `Sequential` | Container for stacking layers |

### ⚡ Activation Functions

| Activation | Description |
|------------|-------------|
| `ReLU` | Rectified Linear Unit: `max(0, x)` |
| `Sigmoid` | Sigmoid function: `1 / (1 + exp(-x))` |
| `Tanh` | Hyperbolic tangent |
| `LeakyReLU` | Leaky ReLU with small negative slope |
| `ELU` | Exponential Linear Unit |
| `Softmax` | Softmax function for multi-class classification |
| `LogSoftmax` | Log of softmax function |
| `Softplus` | Softplus function: `log(1 + exp(x))` |
| `Swish` | Swish activation: `x * sigmoid(x)` |
| `GELU` | Gaussian Error Linear Unit |
| `Identity` | Identity function (no activation) |

### 📉 Loss Functions

| Loss Function | Description | Use Case |
|---------------|-------------|----------|
| `MSELoss` | Mean Squared Error | Regression |
| `CrossEntropyLoss` | Cross Entropy Loss | Multi-class Classification |
| `NLLLoss` | Negative Log Likelihood | Classification |
| `BCELoss` | Binary Cross Entropy | Binary Classification |
| `BCEWithLogitsLoss` | BCE with sigmoid built-in | Binary Classification |
| `HuberLoss` | Huber loss (smooth L1) | Regression |

### 🔧 Optimizers

| Optimizer | Description |
|-----------|-------------|
| `SGD` | Stochastic Gradient Descent |
| `SGDWithMomentum` | SGD with Momentum |
| `NAG` | Nesterov Accelerated Gradient |
| `AdaGrad` | Adaptive Gradient |
| `Adam` | Adaptive Moment Estimation |
| `RMSProp` | Root Mean Square Propagation |

## API Reference

### Creating a Model

#### Method 1: Using Sequential API

```
python
from ann_engine.layers import Sequential, Dense, ReLU, Softmax

# Create a sequential model
model = Sequential([
    Dense(256, activation='relu', input_shape=784),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
```

#### Method 2: Using Model Class (Custom Architecture)

```
python
from ann_engine.engine.models import Model
from ann_engine.layers import Dense, ReLU, Softmax

class MyModel(Model):
    def __init__(self):
        super().__init__()
        self.fc1 = Dense(256, activation='relu')
        self.fc2 = Dense(128, activation='relu')
        self.fc3 = Dense(10, activation='softmax')
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

model = MyModel()
```

### Compiling the Model

```
python
from ann_engine.optimizers import Adam, SGD, RMSProp

# Using Adam optimizer (recommended default)
model.compile(
    loss='cross_entropy',  # or 'mse', 'bce', etc.
    optimizer=Adam(learning_rate=0.001)
)

# Using SGD with momentum
model.compile(
    loss='mse',
    optimizer=SGD(learning_rate=0.01, momentum=0.9)
)

# Using RMSProp
model.compile(
    loss='cross_entropy',
    optimizer=RMSProp(learning_rate=0.001)
)
```

### Training the Model

```
python
import numpy as np

# X_train: (n_samples, n_features)
# y_train: (n_samples,) - class labels for classification
#          or (n_samples, n_outputs) for regression

# Basic training
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Training with validation
model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    
)

# Training with verbose output
model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    verbose=1
)
```

### Making Predictions

```
python
# Predict class labels
predictions = model.predict(X_test)
# Returns: array of class indices

# Get prediction probabilities
probabilities = model.predict_proba(X_test)
# Returns: array of shape (n_samples, n_classes)
```

### Evaluating the Model

```
python
# Evaluate on test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
```

## Examples

### MNIST Classification

```
python
import numpy as np
from ann_engine.layers import Sequential, Dense
from ann_engine.losses import CrossEntropyLoss
from ann_engine.optimizers import Adam

# Load and preprocess MNIST data
# X_train: (60000, 784), y_train: (60000,)
# X_test: (10000, 784), y_test: (10000,)

# Flatten images: (28, 28) -> (784,)
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

# Normalize to [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# Create model
model = Sequential([
    Dense(256, activation='relu', input_shape=784),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile and train
model.compile(
    loss='cross_entropy',
    optimizer=Adam(lr=0.001)
)

model.fit(X_train, y_train, epochs=10, batch_size=128, verbose=1)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
```

### Binary Classification

```
python
from ann_engine.layers import Sequential, Dense, Sigmoid
from ann_engine.losses import BCELoss
from ann_engine.optimizers import Adam

# Create model for binary classification
model = Sequential([
    Dense(64, activation='relu', input_shape=20),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='bce',  # Binary Cross Entropy
    optimizer=Adam(lr=0.001)
)

model.fit(X_train, y_train, epochs=50, batch_size=32)

# Predictions are probabilities between 0 and 1
predictions = model.predict(X_test)
```

### Regression

```
python
from ann_engine.layers import Sequential, Dense, Identity
from ann_engine.losses import MSELoss, HuberLoss
from ann_engine.optimizers import SGD

# Create model for regression
model = Sequential([
    Dense(128, activation='relu', input_shape=13),
    Dense(64, activation='relu'),
    Dense(1, activation='identity')  # No activation for regression
])

model.compile(
    loss='mse',  # or 'huber' for Huber loss
    optimizer=SGD(lr=0.01, momentum=0.9)
)

model.fit(X_train, y_train, epochs=100, batch_size=32)

# Predict continuous values
predictions = model.predict(X_test)
```

### Custom Layer Configuration

```
python
from ann_engine.layers import Dense, ReLU

# Customizing Dense layer parameters
layer = Dense(
    units=512,           # Number of neurons
    activation='relu',   # Activation function
    use_bias=True,       # Whether to use bias
    kernel_initializer='glorot',  # Weight initialization
    bias_initializer='zeros'
)
```

## Advanced Usage

### Learning Rate Scheduling

```
python
from ann_engine.optimizers import Adam

# Using a smaller learning rate
optimizer = Adam(learning_rate=0.0001)
model.compile(loss='cross_entropy', optimizer=optimizer)
```

### Early Stopping (Manual Implementation)

```
python
best_loss = float('inf')
patience = 5
no_improvement_count = 0

for epoch in range(100):
    model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)
    val_loss = model.evaluate(X_val, y_val)[0]
    
    if val_loss < best_loss:
        best_loss = val_loss
        no_improvement_count = 0
        # Save best model weights
        best_weights = model.get_weights()
    else:
        no_improvement_count += 1
        if no_improvement_count >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
```

### Saving and Loading Weights

```
python
# Get model weights
weights = model.get_weights()

# Set model weights
model.set_weights(weights)
```

## Architecture Details

### Forward Propagation

```
Input → Dense (linear) → Activation → ... → Output
```

### Backpropagation

The framework automatically computes gradients through automatic differentiation:

1. Forward pass: Compute outputs and cache intermediate values
2. Backward pass: Compute gradients using chain rule
3. Update weights: Use optimizer to update parameters

### Supported Input Formats

- **Training Data**: `(n_samples, n_features)` - 2D numpy array
- **Labels**: 
  - Classification: `(n_samples,)` - 1D array of class indices
  - Regression: `(n_samples, n_outputs)` - 2D array

## Performance Tips

1. **Batch Size**: Start with 32-128. Larger batches train faster but may require learning rate adjustment.

2. **Learning Rate**: 
   - Adam: 0.001 (default)
   - SGD: 0.01-0.1
   - Adjust based on convergence behavior

3. **Network Architecture**:
   - Start with simpler networks
   - Add complexity gradually
   - Monitor for overfitting

4. **Data Preprocessing**:
   - Normalize/standardize inputs
   - Shuffle data between epochs

## Troubleshooting

### NaN Losses

- Reduce learning rate
- Check for division by zero
- Verify data preprocessing
- Consider gradient clipping

### Poor Convergence

- Increase model capacity
- Adjust learning rate
- Check activation function choices
- Verify data quality

### Memory Issues

- Reduce batch size
- Use smaller network
- Process data in chunks

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Acknowledgments

This project is inspired by:
- PyTorch
- TensorFlow
- Keras

Built with ❤️ using NumPy.

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

## Version History

- **0.1.0**: Initial release
  - Basic Dense and Sequential layers
  - Common activation functions
  - Core optimizers (SGD, Adam, etc.)
  - Loss functions for classification and regression
