from abc import ABC, abstractmethod
from ann_engine.core import Parameter


class Module(ABC):
    def __init__(self):
        self.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def parameters(self):
        """
        Collect all Parameter instances by inspecting instance attributes.
        Works recursively for nested Modules.
        """
        params = []
        seen_ids = set()

        for attr in vars(self):          # ← vars(self) not dir(self)
            obj = getattr(self, attr)
            if isinstance(obj, Parameter):
                if id(obj) not in seen_ids:
                    seen_ids.add(id(obj))
                    params.append(obj)
            elif isinstance(obj, Module):
                for p in obj.parameters():
                    if id(p) not in seen_ids:
                        seen_ids.add(id(p))
                        params.append(p)

        return params

    def train(self):
        """Set module to training mode."""
        self.training = True
        for child in self.children():
            child.train()

    def eval(self):
        """Set module to evaluation mode."""
        self.training = False
        for child in self.children():
            child.eval()

    def children(self):
        """Return direct child Modules."""
        children = []
        for attr in vars(self):
            obj = getattr(self, attr)
            if isinstance(obj, Module):
                children.append(obj)
        return children