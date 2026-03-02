import numpy as np
 
 
class Datasets:
    def __init__(self, n=1000, n_classes=3, noise=0.4):
        self.__n = n
        self.__n_classes = n_classes
        self.__noise = noise
        self.__x = []
        self.__y = []
        self.classes_ = np.arange(n_classes)  # Add sklearn-like attribute
        
    def make_spiral(self):
        per_class = self.__n // self.__n_classes
        for c in range(self.__n_classes):
            t = np.linspace(0, 1, per_class)
            angle = t * 4 * np.pi + (2 * np.pi * c / self.__n_classes)
            r = t
            x1 = r * np.cos(angle) + np.random.randn(per_class) * self.__noise
            x2 = r * np.sin(angle) + np.random.randn(per_class) * self.__noise
            self.__x.append(np.column_stack([x1, x2]))
            self.__y.extend([c] * per_class)
        
        self.__x = np.vstack(self.__x).astype(np.float32)
        self.__y_onehot = np.eye(self.__n_classes)[np.array(self.__y)].astype(np.float32)
        self.__y = np.array(self.__y)  # Store original class labels
        
        idx = np.random.permutation(len(self.__x))
        return self.__x[idx], self.__y_onehot[idx]
    
    @property
    def target(self):
        return self.__y
    
    @property
    def target_onehot(self):
        return self.__y_onehot

