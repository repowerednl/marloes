from .base import BaseAlgorithm
from marloes.networks.WorldModel import WorldModel
from marloes.networks.ActorCritic import ActorCritic
from marloes.data.replaybuffer import ReplayBuffer
from marloes.networks.util import dict_to_tens

import random
from torch.optim import Adam
import torch
import logging


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

    def train(self, update_step: int = 100) -> None:
        """
        Executes the training process for the algorithm.

        This method can be overridden by subclasses for algorithm-specific behavior.
        """
        logging.info("Starting training process...")
        observations, infos = self.environment.reset()
        capacity = 1000 * update_step
        logging.info(f"Initializing ReplayBuffer with capacity {capacity}...")
        RB = ReplayBuffer(capacity=capacity, device=self.device)

        for epoch in range(self.epochs):
            if epoch % 1000 == 0:
                logging.info(f"Reached epoch {epoch}/{self.epochs}...")

            # Gathering experiences
            obs = dict_to_tens(observations, concatenate_all=True)
            actions = self.get_actions(observations)
            observations, rewards, dones, infos = self.environment.step(actions)
            # Add to ReplayBuffer
            acts = dict_to_tens(actions, concatenate_all=True)
            rew = dict_to_tens(rewards, concatenate_all=True)
            rew = torch.tensor([rew.sum()])
            RB.push(obs, acts, rew)

            # For x timesteps, perform the training step on a sample (update_step size) from the ReplayBuffer
            if epoch % update_step == 0 and epoch != 0:
                logging.info("Performing training step...")
                sample = RB.sample(update_step, True)
                dones = torch.ones(update_step)
                # passing artificial dones: a tensor with all continuation (1) flags, except for the last one (0) - length x
                dones[-1] = 0
                self._train_step(
                    sample["obs"],
                    sample["action"],
                    sample["reward"],
                )

            # After chunk is "full", it should be saved
            if self.chunk_size != 0 and epoch % self.chunk_size == 0 and epoch != 0:
                logging.info("Saving intermediate results and resetting extractor...")
                self.saver.save(extractor=self.environment.extractor)
                # clear the extractor
                self.environment.extractor.clear()

        # Save the final results and TODO: model
        logging.info("Training finished. Saving results...")
        self.saver.final_save(self.environment.extractor)

        logging.info("Training process completed.")

    def _train_step(self, obs, actions, rewards, dones):
        """
        Dummy training step, simulating training process. next_obs are probably not used but added for now.
        """
        print("Training step...")
        print(obs.shape)
        print(actions.shape)
        print(rewards.shape)
        print(dones.shape)
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
