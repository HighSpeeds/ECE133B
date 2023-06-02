class kernel:
    """Base class for all kernels.
    """
    def __init__(self):
        pass

    def update_params(self, params_delta: dict):
        """Update the parameters of the kernel.
        """
        for key, value in params_delta.items():
            param = getattr(self, key)
            setattr(self, key, param + value)

    def set_params(self, params: dict):
        """Set the parameters of the kernel.
        """
        for key, value in params.items():
            setattr(self, key, value)

    def get_params(self):
        """Get the parameters of the kernel.
        """
        return {key: getattr(self, key) for key in self.__dict__.keys() if not key.startswith('__')}


