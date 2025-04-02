from .base import BaseAlgorithm
from marloes.networks.WorldModel import WorldModel
from marloes.networks.ActorCritic import ActorCritic
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
        self.world_model = WorldModel(
            observation_shape=self.environment.observation_space,
            action_shape=self.environment.action_space,
        )  # also configurable with HyperParams, using defaults for now

    def _initialize_actor_critic(self):
        """
        Initializes the actor and critic networks.
        """
        input = (
            self.world_model.rssm.hidden_size + self.world_model.rssm.latent_state_size
        )  # h_t + z_t
        self.actor_critic = ActorCritic(
            input=input, output=self.environment.action_space[0]
        )

    def get_actions(self, observations):
        """
        Actions sampled from DreamerV3 ActorCritic need s_t = {h_t, z_t}.
        """
        z_t, _ = self.world_model.encoder(observations)
        h_t = self.world_model.rssm._init_state(
            1
        )  # TODO: this is not correct - how to maintain the recurrent state
        # last layer
        h_t = h_t[-1].squeeze(0).squeeze(0)  # Feels wrong
        s = torch.cat([h_t, z_t], dim=-1)
        actions = self.actor_critic.act(s)
        # return a dict with actions (agent_id: action)
        return {
            agent_id: actions[i]
            for i, agent_id in enumerate(self.environment.agent_dict.keys())
        }

    def _train_step(self, obs, actions, rewards, dones):
        """
        Dummy training step, simulating training process. next_obs are probably not used but added for now.
        """
        # 1. WorldModel Loss
        world_losses = self.world_model.learn(obs, actions, rewards, dones)
        print(world_losses)

        # 2. ActorCritic Loss
        # get the first element of the obs-tensor as the initial state
        initial = obs[0].unsqueeze(0)  # should have shape (1, obs_shape)
        print(initial.shape)
        trajectories = self.world_model.imagine(
            initial=initial, actor=self.actor_critic.actor
        )
        actorcritic_losses = self.actor_critic.learn(trajectories)
        print(actorcritic_losses)

        return
