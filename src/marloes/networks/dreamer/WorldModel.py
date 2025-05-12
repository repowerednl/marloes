import torch
import torch.nn as nn
import torch.nn.functional as F

from .RSSM import RSSM
from marloes.networks.base import BaseNetwork, HyperParams
from .ActorCritic import Actor
from marloes.networks.util import (
    dist,
    symlog_squared_loss,
    gaussian_kl_divergence,
    kl_free_bits,
)

import logging


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
        state_dim: tuple,
        action_dim: tuple,  # Unused now, but added if we want init more dynamically.
        params: dict = None,
        hyper_params: HyperParams = None,
    ):
        """
        Initializes the World Model.

        Args:
            state_dim (tuple): Shape of the observation space.
            action_dim (tuple): Shape of the action space.
            params (dict, optional): Dictionary with network parameters.
            hyper_params (HyperParams, optional): Hyperparameters for the network.
        """
        self.rssm = RSSM(
            x_shape=state_dim[0],
            params=params,
            hyper_params=hyper_params,
            stochastic=True,
        )
        # RSSM in between, is created first to ensure the link between encoder and decoder
        self.decoder = Decoder(self.rssm.fc.out_features, state_dim[0])
        self.reward_predictor = RewardPredictor(
            self.rssm.rnn.hidden_size, self.rssm.fc.out_features
        )
        self.continue_predictor = ContinuePredictor(
            self.rssm.rnn.hidden_size, self.rssm.fc.out_features
        )

        self.modules = [
            self.rssm,
            self.decoder,
            self.reward_predictor,
            self.continue_predictor,
        ]  # TODO: change WorldModel to a nn.Module
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
        self.loss = []

    def imagine(
        self, starting_points: torch.Tensor, actor: Actor, horizon: int = 16
    ) -> list[dict]:
        """
        Generates imagined trajectories from the initial state using the actor.

        Args:
            starting_points (torch.Tensor): Batch of initial states.
            actor (Actor): Actor model to sample actions.
            horizon (int, optional): Length of the imagined trajectory. Defaults to 16.

        Returns:
            list[dict]: List of imagined trajectories containing states, actions, and rewards.
        """
        with torch.no_grad():
            batch = []
            # we have a batch of starting points
            for x in starting_points:
                imagined = {
                    "states": [],
                    "rewards": [],
                    "actions": [],
                }
                x = x.unsqueeze(0)
                # Obtain h_t
                h_0 = self.rssm._init_state(1)
                h_0 = h_0[-1]
                # take the last layer of the GRU, shape (batch=1, hidden_size)
                # infer z_t
                z_0, _ = self.rssm._get_latent_state(h_0)

                # sample action from the actor with model state
                # model state is the concatenation of h_t and z_t
                s = torch.cat([h_0, z_0], dim=-1)

                a_0 = actor(s).sample()  # shape (batch=1, action_dim)

                # should be of shapes [1, length] before passing to rssm
                h_t, z_hat_t, _ = self.rssm.forward(h_0, z_0, a_0)

                # form model state
                x = torch.cat(
                    [x, h_t], dim=-1
                )  # shape (batch=1, obs_dim + hidden_size)
                z_t, _ = self.rssm.encoder(x)
                a_t = a_0
                # Store the initial state/action/reward (?) TODO: yes or no?
                """
                At each starting point, we imagine trajectories of length horizon.
                """
                for t in range(horizon):
                    # get the action from the model state
                    s = torch.cat([h_t, z_t], dim=-1)
                    a_t = actor(s).sample()

                    # Store the imagined states, actions and rewards
                    imagined["states"].append(s)
                    imagined["actions"].append(a_t)

                    # Get h_t from sequence model (transition)
                    h_t, z_t, _ = self.rssm.forward(h_t, z_t, a_t)

                    # Predict the reward
                    r_t = self.reward_predictor(h_t, z_t)

                    imagined["rewards"].append(r_t)

                # Stack the imagined states, actions and rewards
                imagined["states"] = torch.stack(imagined["states"], dim=0)
                imagined["actions"] = torch.stack(imagined["actions"], dim=0)
                imagined["rewards"] = torch.stack(imagined["rewards"], dim=0)
                # Append the imagined sequence to the batch
                batch.append(imagined)
        return batch

    def learn(self, sample: list[dict]) -> dict:
        """
        Performs a learning step for the world model using real trajectories.

        Args:
            sample (list[dict]): Batch of real trajectories.

        Returns:
            dict: Dictionary containing dynamics, representation, prediction, and total losses.
        """
        # extract the to be predicted states (next_states) as tensors into x
        x = torch.stack([sequence["state"] for sequence in sample], dim=0)
        # extract the rewards and done signals as tensors into rew
        rew = torch.stack([sequence["rewards"] for sequence in sample], dim=0)

        # | -----------------------------------------------------------------------------|#
        # | Step 1: Perform rollout in RSSM                                              |#
        # |  - Obtain predicted latent states [h_t-1, z_t-1, a_t-1] -> h_t -> z_hat_t    |#
        # |  - And actual latent states [h_t, x_t] -> z_t                                |#
        # | ---------------------------------------------------------------------------- |#
        results = self.rssm.rollout(sample)
        # returns a dictionary with predicted and actual latent states, recurrent states, and their details

        # | -----------------------------------------------------------------------------|#
        # | Step 2: Predict Reward (and Continue signal)                                 |#
        # |  - Obtain predicted reward [h_t,z_t] -> r_t                                  |#
        # |  - Obtain Continue signal [h_t,z_t] -> c_t    #TODO or TOREMOVE              |#
        # | ---------------------------------------------------------------------------- |#
        h = results["recurrent_states"]
        z = results["actual"]
        r_ts = self.reward_predictor(h, z)

        # | -----------------------------------------------------------------------------|#
        # | Step 3: Decode the predicted latent state to the observations                |#
        # |  - Obtain predicted observations [z] -> x_hat_t                              |#
        # | ---------------------------------------------------------------------------- |#
        x_hat_t = self.decoder(z)

        # | -----------------------------------------------------------------------------|#
        # | Step 4: Compute the loss functions                                           |#
        # |  - L_dyn: KL-divergence between predicted and true latent state              |#
        # |  - L_rep: KL-divergence between predicted and true latent state              |#
        # |  - L_pred: prediction loss (symlog squared loss)                             |#
        # | ---------------------------------------------------------------------------- |#
        # unpack the predicted latent states, and distribution (in details)
        # z_hat = results["predicted"]

        def unpack_details(details: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Takes a list of dictionaries. The dictionary contains the mean and logvar of the latent state.
            Should return mean and logvar (tensors) separately as a batch (lenth of the list).

            Args:
                details (list[dict]): List of dictionaries containing mean and logvar.

            Returns:
                tuple[torch.Tensor, torch.Tensor]: Mean and logvar tensors.
            """
            mean = torch.stack([d["mean"] for d in details], dim=0)
            logvar = torch.stack([d["logvar"] for d in details], dim=0)
            return mean, logvar

        z_hat_mean, z_hat_logvar = unpack_details(results["predicted_details"])
        z_mean, z_logvar = unpack_details(results["actual_details"])
        """
        First loss function, the dynamics loss trains the sequence model to predict the next representation:
        KL-divergence between the predicted latent state and the true latent state (with stop-gradient operator)
                L_dyn = max(1,KL[sg(z_t) || z_hat_t])
        [stop gradient (sg) can be implemented with detach()]
        """
        # dynamic_loss = torch.nn.functional.kl_div(
        #     z_hat, z.detach(), reduction="batchmean"
        # )
        # alternative: KL with free bits (using the mu and logvar from the details gaussian distribution)
        pre_kl = gaussian_kl_divergence(
            z_mean.detach(),
            z_logvar.detach(),
            z_hat_mean,
            z_hat_logvar,
        )
        dynamic_loss = kl_free_bits(kl=pre_kl, free_bits=1.0)

        """
        Second loss function, the representation loss trains the representations to be more predictable
        KL-divergence between the predicted latent state and the true latent state
                L_rep = max(1,KL[z_t || sg(z_hat_t)])
        [stop gradient can be implemented with detach()]
        """
        # representation_loss = torch.nn.functional.kl_div(
        #     z_hat.detach(), z, reduction="batchmean"
        # )
        # alternative: KL with free bits (using the mu and logvar from the details gaussian distribution)
        pre_kl = gaussian_kl_divergence(
            z_hat_mean.detach(),
            z_hat_logvar.detach(),
            z_mean,
            z_logvar,
        )
        representation_loss = kl_free_bits(kl=pre_kl, free_bits=1.0)
        """
        Third loss function, the prediction loss is end-to-end training of the model
        trains the decoder and reward predictor via the symlog squared loss and the continue predictor via logistic regression (not implemented)
        """
        prediction_loss = -symlog_squared_loss(x_hat_t, x) - symlog_squared_loss(
            r_ts, rew
        )

        total_loss = (
            self.beta_weights["dyn"] * dynamic_loss
            + self.beta_weights["rep"] * representation_loss
            + self.beta_weights["pred"] * prediction_loss
        )

        # Backpropagate the total loss
        self.optim.zero_grad()
        total_loss.backward()
        # add gradient clipping # TODO: change WorldModel to a nn.Module
        torch.nn.utils.clip_grad_norm_(
            [param for mod in self.modules for param in mod.parameters()], max_norm=1.0
        )
        self.optim.step()

        # Store the loss in the list
        # self.loss.append(total_loss.item())

        self.loss = {
            "dynamics_loss": dynamic_loss.item(),
            "representation_loss": representation_loss.item(),
            "prediction_loss": prediction_loss.item(),
            "total_world_loss": total_loss.item(),
        }

        # return {
        #     "dynamics_loss": dynamic_loss.item(),
        #     "representation_loss": representation_loss.item(),
        #     "prediction_loss": prediction_loss.item(),
        #     "total_world_loss": total_loss.item(),
        # }


class Decoder(BaseNetwork):
    """
    Decodes the latent state to observations using a simple MLP.
    """

    def __init__(self, latent_dim: int, output: int, hidden_dim: int = 256):
        """
        Initializes the Decoder.

        Args:
            latent_dim (int): Dimension of the latent state.
            output (int): Dimension of the output (observation space).
            hidden_dim (int, optional): Dimension of the hidden layer. Defaults to 256.
        """
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output)

    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Decoder.

        Args:
            z_t (torch.Tensor): Latent state.

        Returns:
            torch.Tensor: Predicted observation.
        """
        x = F.relu(self.fc1(z_t.float()))
        x_hat_t = self.fc2(x)
        return x_hat_t


class RewardPredictor(BaseNetwork):
    """
    Predicts the reward from the latent state using a simple MLP.
    """

    def __init__(self, hidden_dim: int, latent_dim: int):
        """
        Initializes the RewardPredictor.

        Args:
            hidden_dim (int): Dimension of the hidden state.
            latent_dim (int): Dimension of the latent state.
        """
        super(RewardPredictor, self).__init__()
        # simple MLP
        self.fc = nn.Linear(hidden_dim + latent_dim, 1)
        # activation function may be added, using unrestricted output for now

    def forward(self, h_t: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the RewardPredictor.

        Args:
            h_t (torch.Tensor): Hidden state.
            z_t (torch.Tensor): Latent state.

        Returns:
            torch.Tensor: Predicted reward.
        """
        x = torch.cat([h_t, z_t], dim=-1)
        r_t = self.fc(x)
        # v2: use tanh activation function to restrict the output to [-1, 1]
        r_t = torch.tanh(r_t)
        return r_t


class ContinuePredictor(BaseNetwork):
    """
    Predicts whether to continue from the latent state using a binary classification task.
    """

    def __init__(self, hidden_dim: int, latent_dim: int):
        """
        Initializes the ContinuePredictor.

        Args:
            hidden_dim (int): Dimension of the hidden state.
            latent_dim (int): Dimension of the latent state.
        """
        super(ContinuePredictor, self).__init__()
        # simple MLP with sigmoid activation function
        self.fc = nn.Linear(hidden_dim + latent_dim, 1)
        self.classify = nn.Sigmoid()

    def forward(self, h_t: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ContinuePredictor.

        Args:
            h_t (torch.Tensor): Hidden state.
            z_t (torch.Tensor): Latent state.

        Returns:
            torch.Tensor: Probability of continuing.
        """
        x = torch.cat([h_t, z_t], dim=-1)
        c_t = self.fc(x)
        return self.classify(c_t)
