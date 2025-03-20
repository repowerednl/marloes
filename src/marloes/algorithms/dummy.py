from .base import BaseAlgorithm
from marloes.networks import WorldModel
from marloes.networks.util import obs_to_tens
import random
from torch.optim import Adam
import torch.nn as nn


class Dummy(BaseAlgorithm):
    __name__ = "Dummy"

    def __init__(self, config: dict):
        """
        Initializes the Dummy algorithm working with the WorldModel for Dreamer.
        """
        super().__init__(config)
        # initialize the world model
        self._initialize_world_model()
        self._initialize_actor_critic()
        # World Model optimizer
        self.world_optimizer = Adam(
            params=[
                param for mod in self.world_model.modules for param in mod.parameters()
            ],
            lr=0.001,
            weight_decay=1e-4,
        )

        self.mse_loss = nn.MSELoss()

    def _initialize_world_model(self):
        """
        Gets observation shape, and actions shape from the environment.
        TODO: loading is not implemented yet. Only creates new WorldModel.
        """
        obs_shape = (1, 1)  # TODO: get observation shape from environment
        action_shape = (1, 1)  # TODO: get action shape from environment
        # TODO: load parameters in if provided in config (if UID is passed)
        self.world_model = WorldModel(
            obs_shape, action_shape
        )  # also configurable with HyperParams, using defaults for now

    def _initialize_actor_critic(self):
        """
        Initializes the actor and critic networks.
        """

        class ActorCritic:  # TODO: use actual implementation
            def __init__(self):
                self.modules = []

        self.actor_critic = ActorCritic()

    def get_actions(self, observations):
        """
        Random actions for each agent in the environment.
        """
        return {agent_id: random.uniform(-1, 1) for agent_id in observations.keys()}

    def _train_step(self, obs, actions, rewards, next_obs, dones):
        """
        Dummy training step, simulating training process...
        """
        # 1. World Model Loss
        # Step 1: Encode observations to latent state to z_t, get the initial hidden state h_t
        # Step 3: Sequence model predicts the sequence given past actions
        # Step 4:

        # 2. Actor Loss

        # 3. Critic Loss

        # ADDITIONAL INFO:
        # WorldModel optimized with: Prediction loss, Dynamics Loss and Representation Loss
        # where Prediction loss trains Decoder and RewardPredictor via the symlog squared
        # and the ContinuePredictor via logistic regression
        # Dynamics Loss trains the sequence model to predict the next representation by minimizing KL-divergence
        # between the predictor and the next stochastic representation.

        pass
