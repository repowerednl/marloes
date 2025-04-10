import torch
import torch.nn as nn
import torch.nn.functional as F

from marloes.networks.base import BaseNetwork, LayerDetails, HyperParams
from marloes.networks.details import RSSM_LD
from marloes.networks.util import dist


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
        self.latent_state_size = details.hidden["dense"]["out_features"]
        self.hidden_size = details.hidden["recurrent"]["hidden_size"]
        # Initialize the RNN for SEQUENCE MODEL:
        self.rnn = nn.GRU(
            **details.hidden["recurrent"]
        )  # Recurrent states produces h_t

        # DYNAMICS MODEL:
        # Initialize the Deterministic dense layer to predict z_hat
        self.fc = nn.Linear(
            self.hidden_size,
            self.latent_state_size,
        )
        # Initialize the Stochastic dense layers to predict z_hat
        self.fc_mu = nn.Linear(
            self.hidden_size,
            self.latent_state_size,
        )
        self.fc_logvar = nn.Linear(
            self.hidden_size,
            self.latent_state_size,
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
        x = torch.cat([h_t, z_t, a_t], dim=-1).float()
        return self.rnn(x)

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

    def rollout(
        self, sample: list[dict]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs a rollout for each sequence in the batch/sample.
        """
        pred_z = []
        z = []
        h = []
        for sequence in sample:
            states = sequence["state"]
            actions = sequence["actions"]
            next_states = sequence["next_state"]

            # Get the predicted and actual latent states
            predicted, actual, h_ts = self._single_rollout(states, actions, next_states)
            pred_z.append(predicted)
            z.append(actual)
            h.append(h_ts)

        # Return the predicted and actual latent states, and the recurrent states as tensors
        return torch.cat(pred_z, dim=0), torch.cat(z, dim=0), torch.cat(h, dim=0)

    def _single_rollout(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
    ):
        """
        Single rollout for a single sequence in the batch obtaining the predicted and actual latent states.
        """
        h_0 = self._init_state(1)  # Batch size of 1: single rollout
        h_t = h_0[-1]  # Take the last layer and unsqueeze for batch dim

        T = next_states.shape[0]
        # # infer the initial latent state for the first step
        # z_t = self._get_latent_state(h_t)
        # (alternative: Use Encoder instead, since we can use states)
        x = torch.cat([states[0].unsqueeze(0), h_t], dim=-1)
        print("x:", x.shape)
        z_t, _ = self.encoder(x)

        pred_z = []
        z = []
        pred_z_details = []
        z_details = []
        h_ts = []
        for t in range(T):
            a_t = actions[t].unsqueeze(0)  # Add batch dimension
            print("a_t:", a_t.shape)
            print("h_t:", h_t.shape)
            print("z_t:", z_t.shape)
            # ------- STEP 1: Pass through RNN to get h_t and predicted latent state ---------#
            h_t, predicted, predicted_details = self.forward(h_t, z_t, a_t)
            print("h_t:", h_t.shape)
            print("next:", next_states[t].shape)
            h_ts.append(h_t)
            # ------- STEP 2: Use h_t and state information for (actual) latent state through Encoder ---------#
            x = torch.cat([next_states[t].unsqueeze(0), h_t], dim=-1)
            print("x:", x.shape)
            print("states:", next_states[t].shape)
            encoded, encoded_detials = self.encoder(x)
            z_t = encoded
            # ------- STEP 3: Save information ---------#
            pred_z.append(predicted)
            z.append(encoded)
            pred_z_details.append(predicted_details)
            z_details.append(encoded_detials)
        # the details are lists of dictionaries with mean and logvar, return as one dictionary with tensors
        pred_z_details = {
            key: torch.stack([d[key] for d in pred_z_details], dim=0).squeeze(1)
            for key in pred_z_details[0].keys()
        }
        z_details = {
            key: torch.stack([d[key] for d in z_details], dim=0).squeeze(1)
            for key in z_details[0].keys()
        }

        # Return the predicted and actual latent states, and the recurrent states (needed for reward prediction)
        return (
            torch.cat(pred_z, dim=0),
            torch.cat(z, dim=0),
            torch.cat(h_ts, dim=0),
            pred_z_details,
            z_details,
        )

    def add_encoder(self, input: int):
        """
        Adds an encoder to the RSSM network.
        """
        input += self.hidden_size
        self.encoder = Encoder(input, self.latent_state_size)


class Encoder(BaseNetwork):
    """
    Class that encodes the observations to the latent state for the RSSM network.
    Since we have no images (CNN) in this case, we can use a simple MLP.
    """

    def __init__(self, input: int, latent_dim: int, hidden_dim: int = 256):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Passes observations (x) through the MLP to predict latent state.
        """
        x = F.relu(
            self.fc1(x.float())
        )  # float() added to ensure compatibility with torch.tensor float32
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return dist(mu, logvar), {
            "mean": mu,
            "logvar": logvar,
        }
