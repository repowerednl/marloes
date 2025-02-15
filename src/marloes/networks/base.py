import torch


class BaseNetwork(torch.nn.Module):
    """
    Base class for all networks.
    """

    def __init__(self, params=None):
        super(BaseNetwork, self).__init__()
        self.initialize(params)

    def initialize(self, params):
        """
        Initialize the network.
        """
        if params:
            self.load_state_dict(params)
        else:
            raise NotImplementedError
