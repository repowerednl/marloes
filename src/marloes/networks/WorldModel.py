import torch
import torch.nn as nn
import torch.nn.functional as F

from .util import observation_to_tensor
from .RSSM import RSSM


class WorldModel:
    """
    World model using simple MLP Encoder and Decoder for an RSSM network, based on the DreamerV3 architecture.
    Requires:
    - observation shape
    - action shape (for dynamic model)
    """

    def __init__(
        self, observation_shape: tuple, action_shape: tuple, params, hyper_params
    ):
        """
        Initializes the World Model: Encoder (x->z_t) -> RSSM -> Decoder (z_t->x_hat_t)
        TODO: observation_shape is implemented as a tuple, might be better to do conversion from observation (dict) to tensor here.
        """
        self.rssm = RSSM(
            params=params,
            hyper_params=hyper_params,
        )
        self.encoder = Encoder(observation_shape, self.rssm.fc.out_features)
        # RSSM in between, is created first to ensure the link between encoder and decoder
        self.decoder = Decoder(self.rssm.fc.out_features, observation_shape)
        self.reward_predictor = RewardPredictor(
            self.rssm.rnn.hidden_size, self.rssm.fc.out_features
        )
        self.continue_predictor = ContinuePredictor(
            self.rssm.rnn.hidden_size, self.rssm.fc.out_features
        )

    def imagine(self, h_t, z_t, actions):
        """
        Imagine the next states through rollout given a state and a set of actions.
        """
        recurrent_states = []
        preds = []
        # TODO: sequentialCTCE?
        for (
            a_t
        ) in (
            actions
        ):  # TODO: a_t are a set of actions, rssm should be able to handle this
            h_t, z_t = self.rssm(h_t, z_t, a_t)
            recurrent_states.append(h_t)
            preds.append(z_t)
        return torch.stack(preds)

    def forward(self, x, h_t, a_t):
        """
        Forward pass through the networks:
        - Encoding observation (x) to latent state
        - Predict next state through RSSM
        - Decoding latent state to observation
        - predicting actions and value #TODO
        """
        # convert observation to tensor
        x = observation_to_tensor(observation=x, concatenate_all=True)
        # TODO: sequentialCTCE?
        z_t = self.encoder(x)
        h_t, z_hat_t = self.rssm(h_t, z_t, a_t)
        x_hat_t = self.decoder(z_hat_t)
        # step 4
        r_t = self.reward_predictor(h_t, z_hat_t)

        return x_hat_t, z_hat_t, h_t, r_t


class Encoder(nn.Module):
    """
    Class that encodes the observations to the latent state for the RSSM network.
    Since we have no images (CNN) in this case, we can use a simple MLP.
    # TODO: maybe move to RSSM
    """

    def __init__(self, obs_shape: tuple, latent_dim: int, hidden_dim: int = 256):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(obs_shape[0], hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes observations (x) through the MLP to predict latent state.
        """
        x = F.relu(self.fc1(x))
        z_t = self.fc2(x)
        return z_t


class Decoder(nn.Module):
    """
    Class that decodes the latent state to the observations for the RSSM network.
    Since we have no images (CNN) in this case, we can use a simple MLP.
    """

    def __init__(self, latent_dim: int, obs_shape: tuple, hidden_dim: int = 256):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, obs_shape[0])

    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        """
        Receives latent state z_t -> which is a tensor.
        Passes through the MLP to predict the next observation.
        """
        x = F.relu(self.fc1(z_t))
        x_hat_t = self.fc2(x)
        return x_hat_t


class RewardPredictor(nn.Module):
    """
    Class that predicts the reward from the latent state.
    forward pass: (h_t, z_t) -> r_t
    """

    def __init__(self, hidden_dim: int, latent_dim: int):
        super(RewardPredictor, self).__init__()
        # simple MLP
        self.fc = nn.Linear(hidden_dim + latent_dim, 1)
        # activation function may be added, using unrestricted output for now

    def forward(self, h_t: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        """
        Concatenates h_t (size hidden_dim) and z_t (size latent_dim) and predicts the reward.
        """
        x = torch.cat([h_t, z_t], dim=-1)
        r_t = self.fc(x)
        return r_t


class ContinuePredictor(nn.Module):
    """
    Class that predicts whether to continue from the latent state.
    A binary classification task; (h_t, z_t) -> c_t = [0,1].
    """

    def __init__(self, hidden_dim: int, latent_dim: int):
        super(ContinuePredictor, self).__init__()
        # simple MLP with sigmoid activation function
        self.fc = nn.Linear(hidden_dim + latent_dim, 1)
        self.classify = nn.Sigmoid()

    def forward(self, h_t: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        """
        Concatenates h_t (size hidden_dim) and z_t (size latent_dim) and predicts whether to continue.
        """
        x = torch.cat([h_t, z_t], dim=-1)
        c_t = self.fc(x)
        return self.classify(c_t)
