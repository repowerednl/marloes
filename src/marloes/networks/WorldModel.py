import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseNetwork, LayerDetails, HyperParams


class RSSM(BaseNetwork):
    def __init__(self, params=None, layer_details=None, hyper_params=None):
        """
        A Recurrent State Space Model (RSSM) network, based on DreamerV3 architecture.
        """
        # OWN VALIDATION
        # layer details should have a recurrent layer in hidden
        if layer_details.hidden.get("recurrent") is None:
            raise ValueError("Hidden layer must have a recurrent layer.")

        # OWN INITIALIZATION

    def initialize_network(self, params, layer_details):
        """
        Overriding initialization network method.
        """
