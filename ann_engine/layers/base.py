class Module:
    """
    Base class for all layers.
    Holds trainable Parameters.
    """
    def __init__(self):
        self._parameters = {}

    def forward(self, *inputs):
        """
        Define in child class.
        """
        raise NotImplementedError

    def __setattr__(self, name, value):
        """
        Override setattr to track Parameters automatically.
        """
        from ann_engine.core.parameter import Parameter

        if isinstance(value, Parameter):
            if not hasattr(self, "_parameters"):
                super().__setattr__("_parameters", {})
            self._parameters[name] = value
        super().__setattr__(name, value)

    def parameters(self):
        """
        Return all Parameters of this module.
        """
        return list(self._parameters.values())