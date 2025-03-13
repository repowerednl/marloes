import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseNetwork, LayerDetails
from .details import RSSM_LD


class RSSM(BaseNetwork):
    def __init__(self, params=None, hyper_params=None, observation_shape: int = None):
        """
        A Recurrent State Space Model (RSSM) network, based on DreamerV3 architecture.
        layer_details are fixed for this network.
        # TODO: Overrides base network right now, would be nice to use that as a proper base class.
        """
        super().__init__(params, RSSM_LD, hyper_params)

    @staticmethod
    def _validate_rssm(details: LayerDetails):
        # own validation: hidden should have "recurrent": {}
        if "recurrent" not in details["hidden"]:
            raise ValueError("RSSM network requires a recurrent hidden layer.")
        if "dense" not in details["hidden"]:
            raise ValueError("RSSM network requires a dense hidden layer.")
        # should have explicit details
        required_keys = [
            "input_size",
            "hidden_size",
            "num_layers",
            "nonlinearity",
            "bias",
            "batch_first",
            "dropout",
            "bidirectional",
        ]
        for key in required_keys:
            if key not in details["hidden"]["recurrent"]:
                raise ValueError(
                    f"Missing key '{key}' in recurrent hidden layer details."
                )
        required_keys = ["out_features"]  # TODO: add custom dense layer details
        for key in required_keys:
            if key not in details["hidden"]["dense"]:
                raise ValueError(f"Missing key '{key}' in dense hidden layer details.")

    def initialize_network(self, params, details, hyperparams):
        """
        Overrides the base class initialization.
        """
        self._validate_rssm(details)
        # Initialize the hidden layers, using GRU for now #TODO: add "type" option to details - pop it and select rnn based on this
        self.rnn = nn.GRU(**details["hidden"]["recurrent"])
        self.fc = nn.Linear(
            details["hidden"]["recurrent"]["hidden_size"],
            details["hidden"]["dense"]["hidden_size"],
        )

    def forward(self, h_t, z_t, a_t):
        """
        Forward pass through the network, overriding the base class.
        Predicts the next latent state given the previous state and action.
        """
        h_t = self.rnn(torch.cat([h_t, z_t, a_t], dim=-1))
        z_hat_t = self.fc(h_t)
        return h_t, z_hat_t
