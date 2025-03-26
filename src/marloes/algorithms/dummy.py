from .base import BaseAlgorithm
from marloes.networks.WorldModel import WorldModel
import random
from torch.optim import Adam
import torch


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
        self.epochs = 10

    def _initialize_world_model(self):
        """
        Gets observation shape, and actions shape from the environment.
        TODO: loading is not implemented yet. Only creates new WorldModel.
        """
        obs_shape = self.environment.observation_space
        act_shape = self.environment.action_space
        print(f"Observation Space: {obs_shape}")
        print(f"Action Space: {act_shape}")
        self.world_model = WorldModel(
            obs_shape, act_shape
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
        Dummy training step, simulating training process. next_obs are probably not used but added for now.
        """
        # 1. World Model Loss
        d_loss, r_loss = self.world_model.learn(obs, actions, rewards, dones)

        # 2. Actor Loss

        # 3. Critic Loss

        return

    def _train_world_model(self):
        """
        Training the world model with data from the ReplayBuffer.
        """
        pass
