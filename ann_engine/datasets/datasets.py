import numpy as np
from ann_engine.datasets.base import (
    make_spirals,
    make_moons,
    make_circles,
    make_xor,
    make_regression,
    make_imbalanced,
    make_high_dimensional,
)

class Datasets:
    """Collection of complex dataset generators."""
    
    @staticmethod
    def make_spirals(n=1000, n_classes=3, noise=0.4):
        """Generate spiral dataset."""
        return make_spirals(n, n_classes, noise)
    
    @staticmethod
    def make_moons(n=1000, noise=0.15):
        """Generate moons dataset."""
        return make_moons(n, noise)
    
    @staticmethod
    def make_circles(n=1000, noise=0.1, factor=0.4):
        """Generate circles dataset."""
        return make_circles(n, noise, factor)
    
    @staticmethod
    def make_xor(n=1000, noise=0.1):
        """Generate XOR dataset."""
        return make_xor(n, noise)
    
    @staticmethod
    def make_regression(n=1000, noise=0.2):
        """Generate non-linear regression dataset."""
        return make_regression(n, noise)
    
    @staticmethod
    def make_imbalanced(n=1000, imbalance=0.05):
        """Generate imbalanced classification dataset."""
        return make_imbalanced(n, imbalance)
    
    @staticmethod
    def make_high_dimensional(n=800):
        """Generate high-dimensional dataset with noise."""
        return make_high_dimensional(n)