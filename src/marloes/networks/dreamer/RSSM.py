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
        x_shape: int,
        config: dict = {},
        params: dict = None,
        hyper_params: HyperParams = None,
        actions_shape: int = 0,
        stochastic: bool = False,
    ):
        """
        Initializes the RSSM network.

        Args:
            x_shape (int): Shape of the input observations.
            params (dict, optional): Dictionary with network parameters.
            hyper_params (HyperParams, optional): Hyperparameters for the network.
            stochastic (bool, optional): Whether to use stochastic latent states. Defaults to False.
        """
        self.stochastic = stochastic
        super().__init__()
        self.clamp_upper = config.get("clamp_upper", 5)
        self.clamp_lower = config.get("clamp_lower", -5)
        self.action_dim = actions_shape
        self.initialize_network(params, config.get("LayerDetails", RSSM_LD))
        self.encoder = Encoder(
            x_shape + self.hidden_size,
            self.latent_state_size,
            config=config.get("Encoder", {}),
        )

    @staticmethod
    def _validate_rssm(details: LayerDetails):
        """
        Validates the RSSM layer details.

        Args:
            details (LayerDetails): Layer details for the RSSM network.

        Raises:
            ValueError: If required keys are missing in the layer details.
        """
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

    def initialize_network(self, params: dict, details: dict | LayerDetails):
        """
        Initializes the RSSM network.

        Args:
            params (dict): Network parameters.
            details (LayerDetails): Layer details for the RSSM network.
        """
        self.latent_state_size = details["latent_size"]
        self.hidden_size = details["recurrent_size"]
        # Initialize the RNN for SEQUENCE MODEL:
        self.rnn = nn.GRU(
            input_size=self.latent_state_size + self.hidden_size + self.action_dim,
            hidden_size=self.hidden_size,
            batch_first=details["batch_first"],
            num_layers=details["num_layers"],
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

    def forward(self, h_t: torch.Tensor, z_t: torch.Tensor, a_t: torch.Tensor):
        """
        Forward pass through the RSSM network.

        Args:
            h_t (torch.Tensor): Hidden state.
            z_t (torch.Tensor): Latent state.
            a_t (torch.Tensor): Action.

        Returns:
            tuple: Updated hidden state, predicted latent state, and latent state details.
        """
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

        Args:
            h_t (torch.Tensor): Previous hidden state.
            z_t (torch.Tensor): Previous latent state.
            a_t (torch.Tensor): Action.

        Returns:
            tuple: Output and hidden state from the RNN.
        """
        x = torch.cat([h_t, z_t, a_t], dim=-1).float()
        # protect rnn by clamping
        x = torch.clamp(x, min=self.clamp_lower, max=self.clamp_upper)
        return self.rnn(x)

    def _get_latent_state(self, h_t: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Predicts the latent state from the hidden state.

        Args:
            h_t (torch.Tensor): Hidden state..0

        Returns:
            tuple: Latent state and its details (mean and log variance).
        """
        if self.stochastic:

            mu = self.fc_mu(h_t)
            logvar = self.fc_logvar(h_t)
            # nan values with h_t initialized to 0
            mu = torch.nan_to_num(mu, nan=0.0)
            logvar = torch.clamp(
                torch.nan_to_num(logvar, nan=0.0),
                min=self.clamp_lower,
                max=self.clamp_upper,
            )
            z_t = dist(mu, logvar)
            return z_t, {"mean": mu, "logvar": logvar}

        z_t = self.fc(h_t)
        return z_t, {"mean": None, "logvar": None}

    def _init_state(self, batch_size: int, random_init: bool = True):
        """
        Initializes the hidden state for the RNN.

        Args:
            batch_size (int): Batch size.
            random_init (bool, optional): Whether to initialize with small random values. Defaults to True.

        Returns:
            torch.Tensor: Initialized hidden state.
        """
        if random_init:
            return (
                torch.randn(self.rnn.num_layers, batch_size, self.rnn.hidden_size)
                * 0.01
            )
        return torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size)

    def rollout(self, sample: list[dict]) -> dict:
        """
        Performs a rollout for a batch of sequences.

        Args:
            sample (list[dict]): Batch of sequences.

        Returns:
            dict: Dictionary containing predicted and actual latent states, recurrent states, and their details.
        """
        results = {
            "predicted": [],
            "actual": [],
            "recurrent_states": [],
            "predicted_details": [],
            "actual_details": [],
        }
        for sequence in sample:
            # unpack
            states = sequence["state"]
            actions = sequence["actions"]
            next_states = sequence["next_state"]
            # Get the predicted and actual latent states
            (
                predicted,
                actual,
                h_ts,
                predicted_details,
                actual_details,
            ) = self._single_rollout(states, actions, next_states)
            [
                results[key].append(val)
                for key, val in zip(
                    results.keys(),
                    [predicted, actual, h_ts, predicted_details, actual_details],
                )
            ]
        # Convert lists to tensors or dictionaries as needed
        results["predicted"] = torch.stack(results["predicted"], dim=0)
        results["actual"] = torch.stack(results["actual"], dim=0)
        results["recurrent_states"] = torch.stack(results["recurrent_states"], dim=0)
        return results

    def _single_rollout(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
    ):
        """
        Single rollout for a single sequence in the batch obtaining the predicted and actual latent states.

        Args:
            states (torch.Tensor): Observed states.
            actions (torch.Tensor): Actions taken.
            next_states (torch.Tensor): Next observed states.

        Returns:
            tuple: Predicted latent states, actual latent states, recurrent states, and their details.
        """
        h_0 = self._init_state(1)  # Batch size of 1: single rollout
        h_t = h_0[-1]  # Take the last layer and unsqueeze for batch dim

        T = next_states.shape[0]
        # # infer the initial latent state for the first step
        # z_t = self._get_latent_state(h_t)
        # (alternative: Use Encoder instead, since we can use states)
        x = torch.cat([states[0].unsqueeze(0), h_t], dim=-1)
        z_t, _ = self.encoder(x)

        pred_z = []
        z = []
        pred_z_details = []
        z_details = []
        h_ts = []
        for t in range(T):
            a_t = actions[t].unsqueeze(0)  # Add batch dimension

            # ------- STEP 1: Pass through RNN to get h_t and predicted latent state ---------#
            h_t, predicted, predicted_details = self.forward(h_t, z_t, a_t)
            h_ts.append(h_t)
            # ------- STEP 2: Use h_t and state information for (actual) latent state through Encoder ---------#
            x = torch.cat([next_states[t].unsqueeze(0), h_t], dim=-1)
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


class Encoder(BaseNetwork):
    """
    Class that encodes the observations to the latent state for the RSSM network.
    Since we have no images (CNN) in this case, we can use a simple MLP.
    """

    def __init__(
        self, input: int, latent_dim: int, config: dict = {}, hidden_dim: int = 256
    ):
        """
        Initializes the Encoder.

        Args:
            input (int): Dimension of the input.
            latent_dim (int): Dimension of the latent state.
            hidden_dim (int, optional): Dimension of the hidden layer. Defaults to 256.
        """
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.clamp_upper = config.get("clamp_upper", 5)
        self.clamp_lower = config.get("clamp_lower", -5)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Forward pass through the Encoder.

        Args:
            x (torch.Tensor): Input observations.

        Returns:
            tuple: Latent state and its details (mean and log variance).
        """
        x = F.relu(
            self.fc1(x.float())
        )  # float() added to ensure compatibility with torch.tensor float32
        mu = self.fc_mu(x)
        logvar = torch.clamp(
            self.fc_logvar(x), min=self.clamp_lower, max=self.clamp_upper
        )
        return dist(mu, logvar), {
            "mean": mu,
            "logvar": logvar,
        }
