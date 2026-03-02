import numpy as np
from ann_engine.core import Tensor
from ann_engine.layers.base import Module as LayerModule


class Sequential:
    """
    Sequential model that stacks layers sequentially.
    Similar to tf.keras.Sequential

    Because Dense layers now support lazy initialization, you do NOT need to
    specify input_shape on the first layer — the model will build itself
    automatically on the first call to fit() / predict() / forward().

    Example:
        model = Sequential([
            Dense(128, activation='relu'),   # input_shape inferred automatically
            Dense(64,  activation='relu'),
            Dense(10,  activation='softmax'),
        ])
        model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy',
                      metrics=['accuracy'])
        history = model.fit(X_train, y_train, epochs=20, batch_size=32)
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

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def add(self, layer):
        """Append a layer to the model."""
        self.layers.append(layer)
        self.built = False

    def build(self, input_shape):
        """
        Build the model by running a single dummy forward pass.

        input_shape can be:
            - (n_samples, n_features)  e.g. (1000, 784)  ← from x.data.shape
            - (n_features,)            e.g. (784,)
            - int                      e.g. 784
        """
        if self.built:
            return

        # ── Build a (1, n_features) dummy input ──────────────────────
        if isinstance(input_shape, int):
            dummy = Tensor(np.zeros((1, input_shape)))
        elif isinstance(input_shape, (tuple, list)):
            if len(input_shape) == 1:
                # (784,) → (1, 784)
                dummy = Tensor(np.zeros((1, input_shape[0])))
            else:
                # (1000, 784) or (1, 784) → use only 1 sample
                dummy = Tensor(np.zeros((1, input_shape[-1])))
        else:
            dummy = Tensor(np.zeros((1,) + tuple(input_shape)[1:]))

        # ── Trigger lazy init in all layers ──────────────────────────
        x = dummy
        for layer in self.layers:
            x = layer(x)

        self.built = True
        self._input_shape = dummy.data.shape
        self._output_shape = x.data.shape

        # Wire optimizer now that parameters exist
        if self._optimizer is not None:
            self._set_optimizer_parameters()

    def build_with_input(self, input_data):
        """Build the model using a real input array / Tensor."""
        if not isinstance(input_data, Tensor):
            input_data = Tensor(np.array(input_data, dtype=np.float64))

        x = input_data
        for layer in self.layers:
            x = layer(x)

        self.built = True
        self._input_shape = input_data.data.shape
        self._output_shape = x.data.shape

        if self._optimizer is not None:
            self._set_optimizer_parameters()

        return True

    # ------------------------------------------------------------------
    # Optimizer wiring
    # ------------------------------------------------------------------

    def _set_optimizer_parameters(self):
        """Collect all trainable parameters and hand them to the optimizer."""
        parameters = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params = layer.parameters()
                if isinstance(params, list):
                    parameters.extend(params)
                else:
                    parameters.append(params)

        if hasattr(self._optimizer, 'set_parameters'):
            self._optimizer.set_parameters(parameters)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x):
        """Run x through every layer in sequence."""
        if not isinstance(x, Tensor):
            x = Tensor(np.array(x, dtype=np.float64))

        if not self.built:
            self.build(x.data.shape)

        for layer in self.layers:
            x = layer(x)
        return x

    def __call__(self, x):
        return self.forward(x)

    # ------------------------------------------------------------------
    # compile / fit / predict / evaluate
    # ------------------------------------------------------------------

    def compile(self, optimizer, loss, metrics=None):
        """
        Configure the model for training.

        Args:
            optimizer : Optimizer instance, e.g. Adam(lr=0.001)
            loss      : str or callable — 'mse', 'categorical_crossentropy', …
            metrics   : list of str or callable, e.g. ['accuracy']
        """
        self._optimizer = optimizer
        self._loss = self._get_loss(loss)
        self._metrics = [self._get_metric(m) for m in (metrics or [])]

        if self.built:
            self._set_optimizer_parameters()

    def fit(self, x, y, batch_size=32, epochs=10, validation_split=0.2, verbose=1):
        """
        Train the model.

        Args:
            x                : array-like or Tensor, shape (n_samples, n_features)
            y                : array-like or Tensor, shape (n_samples, …)
            batch_size       : int
            epochs           : int
            validation_split : float in [0, 1)
            verbose          : 0 = silent, 1 = progress every 10 % of epochs

        Returns:
            dict  history with keys 'loss', 'val_loss', and metric keys
        """
        if self._loss is None:
            raise RuntimeError("Call model.compile() before model.fit().")

        # ── Convert to Tensor ────────────────────────────────────────
        if not isinstance(x, Tensor):
            x = Tensor(np.array(x, dtype=np.float64))
        if not isinstance(y, Tensor):
            y = Tensor(np.array(y, dtype=np.float64))

        # ── Build / wire optimizer ───────────────────────────────────
        if not self.built:
            self.build(x.data.shape)

        # Re-check in case optimizer was set before build
        if self._optimizer is not None:
            params = []
            for layer in self.layers:
                if hasattr(layer, 'parameters'):
                    p = layer.parameters()
                    params.extend(p if isinstance(p, list) else [p])
            # Only rewire if optimizer's parameter list is empty / None
            needs_wire = False
            if hasattr(self._optimizer, 'parameters'):
                needs_wire = (self._optimizer.parameters is None
                              or len(self._optimizer.parameters) == 0)
            elif hasattr(self._optimizer, 'set_parameters'):
                needs_wire = True
            if needs_wire:
                self._set_optimizer_parameters()

        # ── Train / val split ────────────────────────────────────────
        n_samples = x.data.shape[0]
        n_val     = int(n_samples * validation_split)
        n_train   = n_samples - n_val

        indices       = np.random.permutation(n_samples)
        train_idx     = indices[:n_train]
        val_idx       = indices[n_train:]

        x_train = Tensor(x.data[train_idx])
        y_train = Tensor(y.data[train_idx])
        x_val   = Tensor(x.data[val_idx]) if n_val > 0 else None
        y_val   = Tensor(y.data[val_idx]) if n_val > 0 else None

        # ── History dict ─────────────────────────────────────────────
        history = {'loss': []}
        for i in range(len(self._metrics)):
            history[f'metric_{i}'] = []
        if n_val > 0:
            history['val_loss'] = []
            for i in range(len(self._metrics)):
                history[f'val_metric_{i}'] = []

        log_every = max(1, epochs // 10)

        # ── Epoch loop ───────────────────────────────────────────────
        for epoch in range(epochs):
            epoch_loss    = 0.0
            epoch_metrics = [0.0] * len(self._metrics)
            n_batches     = 0

            # Shuffle training data each epoch
            perm = np.random.permutation(n_train)
            x_train = Tensor(x_train.data[perm])
            y_train = Tensor(y_train.data[perm])

            for i in range(0, n_train, batch_size):
                end = min(i + batch_size, n_train)
                x_batch = Tensor(x_train.data[i:end])
                y_batch = Tensor(y_train.data[i:end])

                # Forward
                y_pred = self.forward(x_batch)
                loss   = self._loss(y_pred, y_batch)

                # Backward
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                epoch_loss += float(loss.data)
                for j, metric_fn in enumerate(self._metrics):
                    epoch_metrics[j] += metric_fn(y_pred, y_batch)

                n_batches += 1

            avg_loss    = epoch_loss / n_batches
            avg_metrics = [m / n_batches for m in epoch_metrics]

            history['loss'].append(avg_loss)
            for j, val in enumerate(avg_metrics):
                history[f'metric_{j}'].append(val)

            # Validation
            val_loss = None
            if n_val > 0 and x_val is not None:
                y_val_pred = self.forward(x_val)
                val_loss   = float(self._loss(y_val_pred, y_val).data)
                history['val_loss'].append(val_loss)
                for j, metric_fn in enumerate(self._metrics):
                    history[f'val_metric_{j}'].append(metric_fn(y_val_pred, y_val))

            # Verbose logging
            if verbose and (epoch % log_every == 0 or epoch == epochs - 1):
                msg = f"Epoch {epoch + 1}/{epochs}  loss: {avg_loss:.4f}"
                if val_loss is not None:
                    msg += f"  val_loss: {val_loss:.4f}"
                metric_names = ['accuracy', 'mae', 'mse']
                for j in range(len(self._metrics)):
                    name = metric_names[j] if j < len(metric_names) else f'metric_{j}'
                    msg += f"  {name}: {avg_metrics[j]:.4f}"
                    if val_loss is not None:
                        msg += f"  val_{name}: {history[f'val_metric_{j}'][-1]:.4f}"
                print(msg)

        return history

    def predict(self, x):
        """Return model predictions for x."""
        if not isinstance(x, Tensor):
            x = Tensor(np.array(x, dtype=np.float64))
        if not self.built:
            self.build(x.data.shape)
        return self.forward(x)

    def evaluate(self, x, y, batch_size=32):
        """
        Evaluate the model on (x, y).

        Returns:
            list: [avg_loss, metric_0, metric_1, …]
        """
        if not isinstance(x, Tensor):
            x = Tensor(np.array(x, dtype=np.float64))
        if not isinstance(y, Tensor):
            y = Tensor(np.array(y, dtype=np.float64))
        if not self.built:
            self.build(x.data.shape)

        n_samples     = x.data.shape[0]
        total_loss    = 0.0
        total_metrics = [0.0] * len(self._metrics)
        n_batches     = 0

        for i in range(0, n_samples, batch_size):
            end     = min(i + batch_size, n_samples)
            x_batch = Tensor(x.data[i:end])
            y_batch = Tensor(y.data[i:end])

            y_pred = self.forward(x_batch)
            total_loss += float(self._loss(y_pred, y_batch).data)
            for j, metric_fn in enumerate(self._metrics):
                total_metrics[j] += metric_fn(y_pred, y_batch)
            n_batches += 1

        avg_loss    = total_loss / n_batches
        avg_metrics = [m / n_batches for m in total_metrics]
        return [avg_loss] + avg_metrics

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self):
      """Print a Keras-style model summary."""
      print("\n" + "=" * 70)
      print("Model: Sequential")
      print("=" * 70)
      print(f"{'Layer (type)':<35} {'Output Shape':<25} {'Param #':<10}")
      print("-" * 70)

      output_shapes = []
    
      # If model is built, we can compute output shapes
      if self.built and hasattr(self, '_input_shape'):
          # Create dummy input with correct shape
          dummy = Tensor(np.zeros(self._input_shape))
          x = dummy
          for layer in self.layers:
              x = layer(x)
              output_shapes.append(x.data.shape)
      else:
          # Try to build with first layer's input_shape
          if self.layers and hasattr(self.layers[0], 'input_shape') and self.layers[0].input_shape:
              input_shape = self.layers[0].input_shape
              if isinstance(input_shape, tuple):
                  dummy = Tensor(np.zeros((1, input_shape[-1])))
              else:
                  dummy = Tensor(np.zeros((1, input_shape)))
              self.build(dummy.data.shape)
            
              # Now compute shapes
              x = dummy
              for layer in self.layers:
                  x = layer(x)
                  output_shapes.append(x.data.shape)
          else:
              print("Model not built yet. Call fit() or forward() first for output shapes.")

      total_params = 0
      for i, layer in enumerate(self.layers):
          layer_name = layer.__class__.__name__

          params = 0
          if hasattr(layer, 'W') and layer.W is not None:
              params += layer.W.data.size
          if hasattr(layer, 'b') and layer.b is not None:
              params += layer.b.data.size
          total_params += params

          out_shape = str(output_shapes[i]) if i < len(output_shapes) else "?"
          print(f"{f'{i}. {layer_name}':<35} {out_shape:<25} {params:<10,}")

      print("=" * 70)
      print(f"Total params:         {total_params:,}")
      print(f"Trainable params:     {total_params:,}")
      print(f"Non-trainable params: 0")
      print("=" * 70)

    # ------------------------------------------------------------------
    # Loss & metric helpers
    # ------------------------------------------------------------------

    def _get_loss(self, loss):
        if callable(loss):
            return loss
        from ann_engine.losses import (MSELoss, BCELoss, CrossEntropyLoss,
                                       NLLLoss, BCEWithLogitsLoss, HuberLoss)
        loss_map = {
            'mse':                    MSELoss,
            'mean_squared_error':     MSELoss,
            'bce':                    BCELoss,
            'binary_crossentropy':    BCELoss,
            'categorical_crossentropy': CrossEntropyLoss,
            'cross_entropy':          CrossEntropyLoss,
            'bce_logits':             BCEWithLogitsLoss,
            'bce_with_logits':        BCEWithLogitsLoss,
            'nll':                    NLLLoss,
            'huber':                  HuberLoss,
        }
        key = loss.lower()
        if key not in loss_map:
            raise ValueError(f"Unknown loss '{loss}'. Choose from {list(loss_map)}")
        return loss_map[key]()

    def _get_metric(self, metric):
        if callable(metric):
            return metric
        metric_map = {
            'accuracy': self._accuracy_metric,
            'mae':      self._mae_metric,
            'mse':      self._mse_metric,
        }
        key = metric.lower()
        if key not in metric_map:
            raise ValueError(f"Unknown metric '{metric}'. Choose from {list(metric_map)}")
        return metric_map[key]

    def _accuracy_metric(self, y_pred, y_true):
        if y_pred.data.ndim > 1 and y_pred.data.shape[1] > 1:
            return np.mean(
                np.argmax(y_pred.data, axis=1) == np.argmax(y_true.data, axis=1)
            )
        return np.mean((y_pred.data > 0.5).astype(np.float32) == y_true.data)

    def _mae_metric(self, y_pred, y_true):
        return float(np.mean(np.abs(y_pred.data - y_true.data)))

    def _mse_metric(self, y_pred, y_true):
        return float(np.mean((y_pred.data - y_true.data) ** 2))