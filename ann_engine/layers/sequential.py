from ann_engine.core import Tensor
from ann_engine.layers.base import Module as LayerModule
import numpy as np

class Sequential:
    """
    Sequential Model that stacks layer sequentially similar to tf.keras.sequential
    
    Example:
     model = Sequential([
            Dense(128, activation='relu', input_shape=(784,)),
            Dense(64, activation='relu'),
            Dense(10, activation='softmax')
        ])
    """  
    
    def __init__(self, layers=None):
        self.layers = []
        self.built = False
        self._loss = None
        self._optimizer = None
        self._metrics = []
        
        if layers is not None:
            for layer in layers:
                self.add(layer)
                
    def add(self, layer):
        """ add layer to the model """
        self.layers.append(layer)
        self.built = False
        
    def build(self, input_shape):
        """ BUild the model by initilizing all layers """
        if self.built:
            return
        
        x = Tensor(np.zeros((1, *input_shape[1:])))
        for layer in self.layers:
            x = layer(x)
            
        self.built = True
        self._input_shape = input_shape
        self._output_shape = x.data.shape 
        
    def forward(self,x):
        """ forward pass through all layers"""
        if not self.built:
            self.build(x.data.shape)
            
        for layer in self.layers:
            x = layer(x)
            
        return x
    
    def __call__(self,x):
        return self.forward(x)
    
    def compile(self, optimizer, loss, metrics=None):
        """
        Configure the model for training
        
        Args:
            optimizer: Optimizer instance (e.g., Adam(lr=0.001))
            loss: Loss function (e.g., 'mse', 'categorical_crossentropy')
            metrics: List of metrics to track (e.g., ['accuracy'])
        """
        
        self. _optimizer = optimizer
        self._loss = self._get_loss(loss)
        
        if metrics :
            self._metrics  = [self._get_metric(m) for m in metrics]
            
    def fit(self, x, y, batch_size=32, epochs=10, validation_split=0.2, verbose=1):
        """
        Train the model
        
        Args:
            x: Training data
            y: Target data
            batch_size: Number of samples per batch
            epochs: Number of epochs to train
            validation_split: Fraction of data to use for validation
            verbose: Verbosity mode (0, 1, or 2)
        
        Returns:
            History object with training history
        """
        # Convert to tensors if needed
        if not isinstance(x, Tensor):
            x = Tensor(x)
        if not isinstance(y, Tensor):
            y = Tensor(y)
        
        # Split validation data
        n_samples = x.data.shape[0]
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val
        
        # Shuffle and split
        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        x_train = Tensor(x.data[train_indices])
        y_train = Tensor(y.data[train_indices])
        x_val = Tensor(x.data[val_indices])
        y_val = Tensor(y.data[val_indices])
        
        history = {
            'loss': [],
            'val_loss': [],
            **{f'metric_{i}': [] for i in range(len(self._metrics))},
            **{f'val_metric_{i}': [] for i in range(len(self._metrics))}
        }
        
        for epoch in range(epochs):
            # Training
            epoch_loss = 0
            epoch_metrics = [0] * len(self._metrics)
            n_batches = 0
            
            # Mini-batch training
            for i in range(0, n_train, batch_size):
                end_idx = min(i + batch_size, n_train)
                x_batch = Tensor(x_train.data[i:end_idx])
                y_batch = Tensor(y_train.data[i:end_idx])
                
                # Forward pass
                y_pred = self.forward(x_batch)
                loss = self._loss(y_pred, y_batch)
                
                # Backward pass
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                
                epoch_loss += loss.data
                
                # Compute metrics
                for j, metric_fn in enumerate(self._metrics):
                    epoch_metrics[j] += metric_fn(y_pred, y_batch)
                
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            avg_metrics = [m / n_batches for m in epoch_metrics]
            
            history['loss'].append(avg_loss)
            for j, val in enumerate(avg_metrics):
                history[f'metric_{j}'].append(val)
            
            # Validation
            if n_val > 0:
                y_val_pred = self.forward(x_val)
                val_loss = self._loss(y_val_pred, y_val).data
                history['val_loss'].append(val_loss)
                
                for j, metric_fn in enumerate(self._metrics):
                    val_metric = metric_fn(y_val_pred, y_val)
                    history[f'val_metric_{j}'].append(val_metric)
            
            # Verbose output
            if verbose and (epoch % max(1, epochs//10) == 0 or epoch == epochs-1):
                print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f}", end='')
                if n_val > 0:
                    print(f" - val_loss: {val_loss:.4f}", end='')
                print()
        
        return history
    
    
                    
                
               
        
        