from .base import BaseAlgorithm
from marloes.networks.dreamer.WorldModel import WorldModel
from marloes.networks.dreamer.ActorCritic import ActorCritic
from marloes.data.replaybuffer import ReplayBuffer
from marloes.networks.util import dict_to_tens

import random
from torch.optim import Adam
import torch
import logging


class Dreamer(BaseAlgorithm):
    __name__ = "Dreamer"

    def __init__(self, config: dict):
        """
        Initializes the Dreamer algorithm working with the WorldModel.
        """
        super().__init__(config)
        self._initialize_world_model()
        self._initialize_actor_critic()
        self.update_interval = self.config.get("update_interval", 100)
        self.previous = None

    def _initialize_world_model(self):
        """
        Initializes the WorldModel with observation and action shapes.
        """
        self.world_model = WorldModel(
            observation_shape=self.environment.observation_space,
            action_shape=self.environment.action_space,
        )

    def _initialize_actor_critic(self):
        """
        Initializes the actor and critic networks.
        """
        input_size = (
            self.world_model.rssm.hidden_size + self.world_model.rssm.latent_state_size
        )
        self.actor_critic = ActorCritic(
            input=input_size, output=self.environment.action_space[0]
        )

    def _init_previous(self):
        """
        Initializes the previous state for the algorithm.
        """
        self.previous = {
            "h_t": self.world_model.rssm._init_state(1),
            "z_t": torch.zeros(1, self.world_model.rssm.latent_state_size),
            "a_t": torch.zeros(1, self.environment.action_space[0]),
        }

    def get_actions(self, observations):
        """
        Computes actions based on the current observations and model state.
        """
        if not self.previous:
            self._init_previous()

        # Step 1: Get the recurrent state (based on previous state)  #
        # ---------------------------------------------------------- #
        h_t, _, _ = self.world_model.rssm(
            self.previous["h_t"], self.previous["z_t"], self.previous["a_t"]
        )
        h_t = h_t[-1].squeeze(0).squeeze(0)

        # Step 2: Get the latent state (based on current obs and h_t)  #
        # ------------------------------------------------------------ #
        x = torch.cat([observations, h_t], dim=-1)
        z_t, _ = self.world_model.encoder(x)

        # Step 3: Get the action (based on the model state)  #
        # -------------------------------------------------- #
        s = torch.cat([h_t, z_t], dim=-1)
        actions = self.actor_critic.act(s)

        return {
            agent_id: actions[i]
            for i, agent_id in enumerate(self.environment.agent_dict.keys())
        }

    def perform_training_steps(self, step: int):
        """
        Executes a training step for the Dreamer algorithm.
        1. The world model is updated with real observations and actions.
        2. The actor-critic model is updated with imagined trajectories and real trajectories.
        """
        if step % self.update_interval != 0:
            return
        # | --------------------------------------------------- |#
        # | Step 1: Get a sample from the replay buffer         |#
        # |  - should be a sample of sequences (size=horizon)   |#
        # | --------------------------------------------------- |#
        real_sample = self.real_RB.sample(self.batch_size)  # sequence = size horizon

        # | ----------------------------------------------------- |#
        # | Step 2: Update the world model with real interactions |#
        # | ----------------------------------------------------- |#
        self.world_model.learn(real_sample)

        # | ----------------------------------------------------- |#
        # | Step 3: Imagine trajectories for ActorCritic learning |#
        # | ----------------------------------------------------- |#

        # | ------------------------------------- |#
        # | Step 4: Update the actor-critic model |#
        # | ------------------------------------- |#
        # Update the critic model with real and imagined trajectories
