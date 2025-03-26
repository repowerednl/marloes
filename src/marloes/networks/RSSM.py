import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseNetwork, LayerDetails, HyperParams
from .details import RSSM_LD
from .util import dist


class RSSM(BaseNetwork):
    """
    A Recurrent State Space Model (RSSM) network, based on DreamerV3 architecture.
    layer_details are fixed for this network.
    Overrides a lot of the base class methods.
    """

    def __init__(
        self,
        params: dict = None,
        hyper_params: HyperParams = None,
        stochastic: bool = False,
    ):
        self.stochastic = stochastic
        super().__init__()
        self.initialize_network(params, RSSM_LD)

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

    def initialize_network(self, params: dict, details: LayerDetails):
        """
        Overrides the base class initialization.
        #TODO: add "type" option to details (recurrent and dense), requires changes in validation
        """
        self._validate_rssm(details)
        # Initialize the RNN for SEQUENCE MODEL:
        self.rnn = nn.GRU(
            **details.hidden["recurrent"]
        )  # Recurrent states produces h_t

        # DYNAMICS MODEL:
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

    def forward(self, h_t: torch.Tensor, z_t: torch.Tensor, a_t: torch.Tensor):
        """
        Forward pass through the network, overriding the base class.
        Predicts the next latent state given the previous state and action.
        Used for one single step in the environment.
        """
        assert (
            torch.cat([h_t, z_t, a_t], dim=-1).shape[-1] == self.rnn.input_size
        ), "RSSM_LD is not configured correctly. Combined input size does not match the RNN input size."
        _, hidden = self._get_recurrent_state(h_t, z_t, a_t)

        # h_t should have the right shape to be passed to the next step
        h_t = hidden[-1].unsqueeze(0)

        # Predict the latent state from the hidden state
        prior, prior_details = self._get_latent_state(h_t)

        return (
            h_t,
            prior,
            prior_details,
        )

    def _get_recurrent_state(
        self, h_t: torch.Tensor, z_t: torch.Tensor, a_t: torch.Tensor
    ):
        """
        Predicts the next hidden state from the previous hidden state, latent state, and action.
        """
        return self.rnn(torch.cat([h_t, z_t, a_t], dim=-1))

    def _get_latent_state(self, h_t: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Predicts the latent state from the hidden state, method depends on stochasticity.
        """
        if self.stochastic:
            mu = self.fc_mu(h_t)
            logvar = self.fc_logvar(h_t)
            z_t = dist(mu, logvar)
            return z_t, {"mean": mu, "logvar": logvar}

        z_t = self.fc(h_t)
        return z_t, {"mean": None, "logvar": None}

    def _init_state(self, batch_size: int):
        """
        Initializes the hidden state for the RNN.
        """
        return torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size)

    def rollout(self, posteriors, actions, dones) -> tuple[list, list]:
        """
        Returns the predicted latent states (priors) for the entire rollout.
        """
        h_t = self._init_state(posteriors.shape[0])  # should be batch size
        priors = []
        priors_details = []
        for z_t, a_t, d in zip(posteriors, actions, dones):
            # Getting the predicted latent states from the network, which requires h_t, z_t and a_T
            h_t, prior, prior_details = self.forward(h_t, z_t, a_t)
            priors.append(prior)
            priors_details.append(prior_details)
        return priors, priors_details
