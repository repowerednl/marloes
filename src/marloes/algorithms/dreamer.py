from .base import BaseAlgorithm
from marloes.networks.WorldModel import WorldModel
from marloes.networks.ActorCritic import ActorCritic
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
        self.epochs = config.get("epochs", 10)
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

        h_t, _, _ = self.world_model.rssm(
            self.previous["h_t"], self.previous["z_t"], self.previous["a_t"]
        )
        h_t = h_t[-1].squeeze(0).squeeze(0)

        x = torch.cat([observations, h_t], dim=-1)
        z_t, _ = self.world_model.encoder(x)

        s = torch.cat([h_t, z_t], dim=-1)
        actions = self.actor_critic.act(s)

        return {
            agent_id: actions[i]
            for i, agent_id in enumerate(self.environment.agent_dict.keys())
        }

    def train(self, update_step: int = 100) -> None:
        """
        Executes the training process for the Dreamer algorithm.
        """
        logging.info("Starting training process...")
        observations, infos = self.environment.reset()
        capacity = 1000 * update_step
        logging.info(f"Initializing ReplayBuffer with capacity {capacity}...")
        RB = ReplayBuffer(capacity=capacity, device=self.device)

        for epoch in range(self.epochs):
            if epoch % 1000 == 0:
                logging.info(f"Reached epoch {epoch}/{self.epochs}...")

            obs = dict_to_tens(observations, concatenate_all=True)
            actions = self.get_actions(observations)
            observations, rewards, dones, infos = self.environment.step(actions)

            acts = dict_to_tens(actions, concatenate_all=True)
            rew = dict_to_tens(rewards, concatenate_all=True)
            rew = torch.tensor([rew.sum()])
            RB.push(obs, acts, rew)

            if epoch % update_step == 0 and epoch != 0:
                logging.info("Performing training step...")
                sample = RB.sample(update_step, True)
                dones = torch.ones(update_step)
                dones[-1] = 0
                self._train_step(
                    sample["obs"],
                    sample["action"],
                    sample["reward"],
                )

            if self.chunk_size != 0 and epoch % self.chunk_size == 0 and epoch != 0:
                logging.info("Saving intermediate results and resetting extractor...")
                self.saver.save(extractor=self.environment.extractor)
                self.environment.extractor.clear()

        logging.info("Training finished. Saving results...")
        self.saver.final_save(self.environment.extractor)
        logging.info("Training process completed.")

    def _train_step(self, obs, actions, rewards, dones):
        """
        Executes a single training step for the Dreamer algorithm.
        """
        logging.info("Training step...")
        world_losses = self.world_model.learn(obs, actions, rewards, dones)
        logging.info(f"WorldModel losses: {world_losses}")

        initial = obs[0].unsqueeze(0)
        trajectories = self.world_model.imagine(
            initial=initial, actor=self.actor_critic.actor
        )
        actorcritic_losses = self.actor_critic.learn(trajectories)
        logging.info(f"ActorCritic losses: {actorcritic_losses}")
