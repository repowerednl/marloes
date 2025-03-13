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
            observation_shape=observation_shape,
        )
        self.encoder = Encoder(observation_shape, self.rssm.rnn.hidden_size)
        self.decoder = Decoder(self.rssm.rnn.hidden_size, observation_shape)

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
        - Encoding observation to latent state
        - Predict next state through RSSM
        - Decoding latent state to observation
        - predicting actions and value #TODO
        """
        # TODO: sequentialCTCE?
        z_t = self.encoder(x)
        h_t, z_hat_t = self.rssm(h_t, z_t, a_t)
        x_hat_t = self.decoder(z_hat_t)
        # step 4

        return x_hat_t, z_hat_t, h_t


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

    def forward(self, x):
        """
        Receives observations x -> which is list of dictionaries.
        First transform the observations to
        """
        x = observation_to_tensor(x)
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

    def forward(self, z_t):
        x = F.relu(self.fc1(z_t))
        x_hat_t = self.fc2(x)
        return x_hat_t


class RewardPredictor(nn.Module):
    """
    Class that predicts the reward from the latent state.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 256):
        pass


class ContinuePredictor(nn.Module):
    """
    Class that predicts whether to continue from the latent state.
    A binary classification task on h_t and z_t.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 256):
        pass
