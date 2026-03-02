import numpy as np


def make_spirals(n=1000, n_classes=3, noise=0.4):
    """Interleaved Archimedean spirals — very non-linear boundary."""
    X, y = [], []
    per_class = n // n_classes
    for c in range(n_classes):
        t = np.linspace(0, 1, per_class)
        angle = t * 4 * np.pi + (2 * np.pi * c / n_classes)
        r = t
        x1 = r * np.cos(angle) + np.random.randn(per_class) * noise * 0.15
        x2 = r * np.sin(angle) + np.random.randn(per_class) * noise * 0.15
        X.append(np.column_stack([x1, x2]))
        y.extend([c] * per_class)
    X = np.vstack(X).astype(np.float32)
    y = np.eye(n_classes)[np.array(y)].astype(np.float32)
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def make_moons(n=1000, noise=0.15):
    """Two interleaving half-moon shapes."""
    n_each = n // 2
    t = np.linspace(0, np.pi, n_each)
    X0 = np.column_stack([np.cos(t), np.sin(t)])
    X1 = np.column_stack([1 - np.cos(t), 1 - np.sin(t) - 0.5])
    X = np.vstack([X0, X1]).astype(np.float32)
    X += np.random.randn(*X.shape).astype(np.float32) * noise
    y = np.array([0]*n_each + [1]*n_each, dtype=np.float32).reshape(-1, 1)
    idx = np.random.permutation(n)
    return X[idx], y[idx]


def make_circles(n=1000, noise=0.1, factor=0.4):
    """Two concentric circles."""
    n_each = n // 2
    t = np.linspace(0, 2 * np.pi, n_each)
    X_inner = np.column_stack([np.cos(t) * factor, np.sin(t) * factor])
    X_outer = np.column_stack([np.cos(t), np.sin(t)])
    X = np.vstack([X_inner, X_outer]).astype(np.float32)
    X += np.random.randn(*X.shape).astype(np.float32) * noise
    y = np.array([0]*n_each + [1]*n_each, dtype=np.float32).reshape(-1, 1)
    idx = np.random.permutation(n)
    return X[idx], y[idx]


def make_xor(n=1000, noise=0.1):
    """XOR pattern — 4 quadrants, alternating labels."""
    X = np.random.uniform(-1, 1, (n, 2)).astype(np.float32)
    y = ((X[:, 0] * X[:, 1]) > 0).astype(np.float32).reshape(-1, 1)
    X += np.random.randn(*X.shape).astype(np.float32) * noise
    return X, y


def make_regression(n=1000, noise=0.2):
    """
    y = sin(x1) * cos(x2) + x3^2 - x4*x5 + noise
    Non-linear with feature interactions.
    """
    X = np.random.randn(n, 8).astype(np.float32)
    y = (np.sin(X[:, 0]) * np.cos(X[:, 1])
         + X[:, 2]**2
         - X[:, 3] * X[:, 4]
         + 0.5 * X[:, 5]).astype(np.float32)
    y += np.random.randn(n).astype(np.float32) * noise
    y = y.reshape(-1, 1)
    # Normalize y to [-1, 1] for stable training
    y = (y - y.mean()) / (y.std() + 1e-8)
    return X, y


def make_imbalanced(n=1000, imbalance=0.05):
    """
    4-class problem where class 0 is very rare (5% of data).
    Tests whether the model can still learn minority class.
    """
    counts = [int(n * imbalance), int(n * 0.35),
              int(n * 0.35), int(n * (1 - imbalance - 0.70))]
    X, y = [], []
    centers = [(-2, -2), (-2, 2), (2, -2), (2, 2)]
    for c, (cnt, (cx, cy)) in enumerate(zip(counts, centers)):
        xi = np.random.randn(cnt, 2) * 0.5 + np.array([cx, cy])
        X.append(xi.astype(np.float32))
        y.extend([c] * cnt)
    X = np.vstack(X)
    y = np.eye(4)[np.array(y)].astype(np.float32)
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def make_high_dimensional(n=800):
    """
    50 features:
      - 5 are truly informative
      - 10 are linear combinations of the informative ones (redundant)
      - 35 are pure noise
    Binary classification.
    """
    X_info = np.random.randn(n, 5).astype(np.float32)
    X_red = (X_info @ np.random.randn(5, 10)).astype(np.float32)
    X_noise = np.random.randn(n, 35).astype(np.float32)
    X = np.hstack([X_info, X_red, X_noise])

    # Label based only on informative features
    score = X_info[:, 0] - X_info[:, 1] + X_info[:, 2] * X_info[:, 3]
    y = (score > 0).astype(np.float32).reshape(-1, 1)
    idx = np.random.permutation(n)
    return X[idx], y[idx]