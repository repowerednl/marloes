import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseNetwork, LayerDetails, HyperParams
from .details import RSSM_LD


class RSSM(BaseNetwork):
    """
    A Recurrent State Space Model (RSSM) network, based on DreamerV3 architecture.
    layer_details are fixed for this network.
    Overrides a lot of the base class methods.
    """

    def __init__(
        self, params=None, hyper_params: HyperParams = None, stochastic: bool = False
    ):
        self.stochastic = stochastic
        super().__init__(params, RSSM_LD, hyper_params)

    @staticmethod
    def _validate_rssm(details: LayerDetails):
        # own validation: hidden should have "recurrent": {}
        if "recurrent" not in details.hidden:
            raise ValueError("RSSM network requires a recurrent hidden layer.")
        if "dense" not in details.hidden:
            raise ValueError("RSSM network requires a dense hidden layer.")
        # should have explicit details
        required_keys = [
            "input_size",
            "hidden_size",
            "num_layers",
            "bias",
            "batch_first",
            "dropout",
            "bidirectional",
        ]
        for key in required_keys:
            if key not in details.hidden["recurrent"]:
                raise ValueError(
                    f"Missing key '{key}' in recurrent hidden layer details."
                )
        required_keys = ["out_features"]  # TODO: add custom dense layer details
        for key in required_keys:
            if key not in details.hidden["dense"]:
                raise ValueError(f"Missing key '{key}' in dense hidden layer details.")

    def initialize_network(self, params, details: LayerDetails):
        """
        Overrides the base class initialization.
        #TODO: add "type" option to details (recurrent and dense), requires changes in validation
        """
        self._validate_rssm(details)
        # Initialize the RNN
        self.rnn = nn.GRU(
            **details.hidden["recurrent"]
        )  # Recurrent states produces h_t

        # Initialize the Deterministic dense layer to predict z_hat
        self.fc = nn.Linear(
            details.hidden["recurrent"]["hidden_size"],
            details.hidden["dense"]["out_features"],
        )
        # Initialize the Stochastic dense layers to predict z_hat
        self.fc_mu = nn.Linear(
            details.hidden["recurrent"]["hidden_size"],
            details.hidden["dense"]["out_features"],
        )
        self.fc_logvar = nn.Linear(
            details.hidden["recurrent"]["hidden_size"],
            details.hidden["dense"]["out_features"],
        )

        if params:
            self._load_from_params(params)
        elif details.random_init:
            self._initialize_random_params()

    def _reparametrize(self, mu, logvar):
        """
        Reparametrization trick to create a stochastic latent state.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, h_t: torch.Tensor, z_t: torch.Tensor, a_t: torch.Tensor):
        """
        Forward pass through the network, overriding the base class.
        Predicts the next latent state given the previous state and action.
        """
        assert (
            torch.cat([h_t, z_t, a_t], dim=-1).shape[-1] == self.rnn.input_size
        ), "RSSM_LD is not configured correctly. Combined input size does not match the RNN input size."
        output, hn = self.rnn(torch.cat([h_t, z_t, a_t], dim=-1))
        # output is of shape (seq_len, batch, num_directions * hidden_size)
        # hn is of shape (num_layers * num_directions, batch, hidden_size)
        h_t = output[:, -1, :].unsqueeze(0)

        # Predict the next latent state (stochastic or deterministic)
        if self.stochastic:
            mu = self.fc_mu(h_t)
            logvar = self.fc_logvar(h_t)
            z_hat_t = self._reparametrize(mu, logvar)
        else:
            z_hat_t = self.fc(h_t)  # Predicts the next latent state

        return h_t, z_hat_t
