import torch
import torch.nn as nn
import torch.nn.functional as F

from .RSSM import RSSM
from .base import BaseNetwork, HyperParams
from .util import dist


def symlog_squared_loss(x, y):
    """
    Symlog squared loss function for Prediction loss in the World Model.
    """

    def symlog(x):
        return torch.sign(x) * torch.log1p(torch.abs(x))

    return F.mse_loss(symlog(x), symlog(y))


class WorldModel:
    """
    World model using simple MLP Encoder and Decoder for an RSSM network, based on the DreamerV3 architecture.
    Requires:
    - observation shape
    - action shape
    Optional:
    - params: dictionary with network parameters
    - hyper_params: HyperParams object with network hyperparameters
    """

    def __init__(
        self,
        observation_shape: tuple,
        action_shape: tuple,  # Unused now, but added if we want init more dynamically.
        params: dict = None,
        hyper_params: HyperParams = None,
    ):
        """
        Initializes the World Model: Encoder (x->z_t) -> RSSM -> Decoder (z_hat_t->x_hat_t)
        """
        self.rssm = RSSM(
            params=params,
            hyper_params=hyper_params,
            stochastic=True,
        )
        self.encoder = Encoder(observation_shape[0], self.rssm.fc.out_features)
        # RSSM in between, is created first to ensure the link between encoder and decoder
        self.decoder = Decoder(self.rssm.fc.out_features, observation_shape[0])
        self.reward_predictor = RewardPredictor(
            self.rssm.rnn.hidden_size, self.rssm.fc.out_features
        )
        self.continue_predictor = ContinuePredictor(
            self.rssm.rnn.hidden_size, self.rssm.fc.out_features
        )

        self.modules = [
            self.rssm,
            self.encoder,
            self.decoder,
            self.reward_predictor,
            self.continue_predictor,
        ]
        # Optimizer
        self.optim = torch.optim.Adam(
            params=[param for mod in self.modules for param in mod.parameters()],
            lr=0.001,
            weight_decay=1e-4,
        )
        self.beta_weights = {
            "pred": 1.0,
            "dyn": 1.0,
            "rep": 0.1,
        }

    def imagine(self, x, actor):
        """
        Imagine function for rollouts from the initial state, using the actor to sample actions (from current policy).
        """
        pass

    def learn(self, obs, act, rew, dones):
        """
        Learning step takes a batch of observations, actions and rewards and dones
        """
        x = obs
        # Posteriors are the latent states from the Encoder (doing this in a batch)
        z_posteriors, post_details = self.encoder(x)
        # Priors are the predicted latent states from the RSSM (doing this in rollout, one by one for the recurrent state)
        z_priors, prior_details, h_ts = self.rssm.rollout(z_posteriors, act)

        # Determine the predicted observation from the latent state
        x_hat_t = self.decoder(z_posteriors)

        h_ts = h_ts.squeeze(1).squeeze(1)

        # Use the RewardPredictor and ContinuePredictor to predict the reward and continue signal
        r_ts = self.reward_predictor(h_ts, z_posteriors)
        c_ts = self.continue_predictor(h_ts, z_posteriors)

        # First loss function, the dynamics loss trains the sequence model to predict the next representation as follows:
        # KL-divergence between the predicted latent state and the true latent state (with stop-gradient operator)
        # L_dyn = max(1,KL[sg(z_t) || z_hat_t]) #### stop gradient can be implemented with detach() on mu and log_var ####
        dynamic_loss = torch.nn.functional.kl_div(
            z_priors, z_posteriors.detach(), reduction="batchmean"
        )
        # Second loss function, the representation loss trains the representations to be more predictable
        # KL-divergence between the predicted latent state and the true latent state
        # L_rep = max(1,KL[z_t || sg(z_hat_t)]) #### stop gradient can be implemented with detach() on mu and log_var ####
        representation_loss = torch.nn.functional.kl_div(
            z_priors.detach(), z_posteriors, reduction="batchmean"
        )

        # Third loss function, the prediction loss is end-to-end training of the model
        # trains the decoder and reward predictor via the symlog squared loss and the continue predictor via logistic regression
        prediction_loss = (
            -symlog_squared_loss(x_hat_t, x)
            - symlog_squared_loss(r_ts.squeeze(-1), rew)
            - F.binary_cross_entropy(c_ts.squeeze(-1), dones)
        )

        total_loss = (
            self.beta_weights["dyn"] * dynamic_loss
            + self.beta_weights["rep"] * representation_loss
            + self.beta_weights["pred"] * prediction_loss
        )
        return dynamic_loss, representation_loss, prediction_loss, total_loss


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


class Decoder(BaseNetwork):
    """
    Class that decodes the latent state to the observations for the RSSM network.
    Since we have no images (CNN) in this case, we can use a simple MLP.
    """

    def __init__(self, latent_dim: int, output: int, hidden_dim: int = 256):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output)

    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        """
        Receives latent state z_t -> which is a tensor.
        Passes through the MLP to predict the next observation.
        """
        x = F.relu(self.fc1(z_t.float()))
        x_hat_t = self.fc2(x)
        return x_hat_t


class RewardPredictor(BaseNetwork):
    """
    Class that predicts the reward from the latent state.
    forward pass: (h_t, z_t (posterior)) -> r_t
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


class ContinuePredictor(BaseNetwork):
    """
    Class that predicts whether to continue from the latent state.
    A binary classification task; (h_t, z_t (posterior)) -> c_t = [0,1].
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
