from .base import BaseAlgorithm
from marloes.networks.dreamer.WorldModel import WorldModel
from marloes.networks.dreamer.ActorCritic import ActorCritic
from marloes.networks.dreamer.ActorCritic2 import ActorCritic2

import torch
import numpy as np

import logging


class Dreamer(BaseAlgorithm):
    """
    Dreamer algorithm implementation using the WorldModel and ActorCritic modules.
    """

    __name__ = "Dreamer"

    def __init__(self, config: dict, evaluate: bool = False):
        """
        Initializes the Dreamer algorithm.

        Args:
            config (dict): Configuration dictionary for the algorithm.
        """
        super().__init__(config, evaluate)
        self.deterministic = self.config.get("deterministic", False)

        self._initialize_world_model(self.config.get("WorldModel", {}))
        self._initialize_actor_critic(self.config.get("ActorCritic", {}))
        self.update_interval = self.config.get("update_interval", 100)
        self.previous = None
        self.horizon = self.config.get("horizon", 16)
        self.losses = None
        self.update_steps = self.config.get("update_steps", 1)
        self.ratio_real_imagined = self.config.get("ratio_real_imagined", 0)

    def _initialize_world_model(self, config: dict):
        """
        Initializes the WorldModel with observation and action shapes.
        """
        self.world_model = WorldModel(
            state_dim=self.environment.state_dim,
            action_dim=self.environment.action_dim,
            config=config,
        )

    def _initialize_actor_critic(self, config: dict):
        """
        Initializes the actor and critic networks.
        """
        input_size = (
            self.world_model.rssm.hidden_size + self.world_model.rssm.latent_state_size
        )
        if not config.get("new", False):
            self.actor_critic = ActorCritic(
                input=input_size,
                output=self.environment.action_dim[0],
                config=config,
                deterministic=self.deterministic,
            )
        else:
            logging.info(
                f"Using ActorCritic2 for Dreamer with input size {input_size} and output size {self.environment.action_dim[0]}"
            )
            self.actor_critic = ActorCritic2(
                input=input_size,
                output=self.environment.action_dim[0],
                config=config,
                deterministic=self.deterministic,
            )

    def _init_previous(self):
        """
        Initializes the previous state for the algorithm.
        """
        self.previous = {
            "h_t": self.world_model.rssm._init_state(1)[-1],
            "z_t": torch.zeros(1, self.world_model.rssm.latent_state_size),
            "a_t": torch.zeros(1, self.environment.action_dim[0]),
        }

    def track_networks(self):
        # should create a dictionary with the networks and their state_dicts
        self.networks = {
            "world_model": self.world_model.state_dict(),
            "world_model_optimizer": self.world_model.optim.state_dict(),
            "actor": self.actor_critic.actor.state_dict(),
            "actor_optimizer": self.actor_critic.actor_optim.state_dict(),
            "critic": self.actor_critic.critic.state_dict(),
            "critic_optimizer": self.actor_critic.critic_optim.state_dict(),
            "critic_target": self.actor_critic.critic_target.state_dict(),
        }

    def get_actions(self, observations: dict, deterministic: bool = False) -> dict:
        """
        Computes actions based on the current observations and model state.

        Args:
            observations (dict): Current observations.

        Returns:
            dict: Dictionary mapping agent IDs to actions.
        """
        if not self.previous:
            self._init_previous()
        # convert observations to tensor
        observations = torch.stack([self.real_RB.dict_to_tens(observations)]).to(
            self.device
        )

        # set world_model to eval mode
        self.world_model.eval()
        # set actor_critic to eval mode
        self.actor_critic.eval()
        with torch.no_grad():
            # Step 1: Get the recurrent state (based on previous state)  #
            # ---------------------------------------------------------- #
            h_t, _, _ = self.world_model.rssm.forward(
                self.previous["h_t"], self.previous["z_t"], self.previous["a_t"]
            )
            h_t = h_t[-1].unsqueeze(0)  # get the last hidden state

            # Step 2: Get the latent state (based on current obs and h_t)  #
            # ------------------------------------------------------------ #
            x = torch.cat([observations, h_t], dim=-1)
            z_t, _ = self.world_model.rssm.encoder(x)

            # Step 3: Get the action (based on the model state)  #
            # -------------------------------------------------- #
            s = torch.cat([h_t, z_t], dim=-1)
            actions, _ = self.actor_critic.act(s, deterministic)

            # Step 4: Update the previous state with the current state  #
            # -------------------------------------------------- #
            self.previous["h_t"] = h_t
            self.previous["z_t"] = z_t
            self.previous["a_t"] = actions

        action_list = actions.squeeze(0).tolist()
        return {
            agent_id: action_list[i]
            for i, agent_id in enumerate(self.environment.trainable_agent_dict.keys())
        }, self.previous

    def perform_training_steps(self, step: int):
        """
        Executes a training step for the Dreamer algorithm.

        Args:
            step (int): Current training step.

        Returns:
            dict: Dictionary containing world model and actor-critic losses.
        """
        # only update when step % update_interval == 0, early return with the (previous) losses
        if step % self.update_interval != 0 or step == 0:  # TODO: uncomment
            return self.losses

        # set world_model to train mode
        self.world_model.train()
        # set actor_critic to train mode
        self.actor_critic.train()

        for _ in range(self.update_steps):
            # | --------------------------------------------------- |#
            # | Step 1: Get a sample from the replay buffer         |#
            # |  - should be a sample of sequences (size=horizon)   |#
            # | --------------------------------------------------- |#
            real_sample = self.real_RB.sample(self.batch_size, self.horizon)

            # | ----------------------------------------------------- |#
            # | Step 2: Update the world model with real interactions |#
            # | ----------------------------------------------------- |#
            self.world_model.learn(real_sample)

            # | ----------------------------------------------------- |#
            # | Step 3: Imagine trajectories for ActorCritic learning |#
            # |  - Sample starting point from the replay buffer       |#
            # |  - Pass Actor to the imagine function in WorldModel   |#
            # |  - Should return a batch of imagined sequences        |#
            # | ----------------------------------------------------- |#
            starting_points = self.real_RB.sample(self.batch_size)
            # only select the ratio
            n_real = int(self.ratio_real_imagined * self.batch_size)
            n_imag = self.batch_size - n_real
            # select batch_size - n_real starting points to pass to imagine
            state_starts = starting_points["state"][:n_imag]
            belief_starts = (
                starting_points["belief"][:n_imag]
                if "belief" in starting_points
                else {}
            )

            imagined_sequences = self.world_model.imagine(
                state_starts,  # starting_points["state"]
                belief_starts,  # starting_points["belief"]
                self.actor_critic.actor,
                self.horizon,
            )

            # | ------------------------------------- |#
            # | Step 4: Update the actor-critic model |#
            # | ------------------------------------- |#
            # Only update with imagined trajectories
            # (updating with real trajectories should be implemented for:
            # - environments where the reward is tricky to predict)
            # states (latent states), actions, rewards of the real trajectory
            # make sure real_sample is converted to the right format (as returned by imagine)
            seq = (
                imagined_sequences
                + self.world_model.convert_samples(real_sample[:n_real])
                if n_real > 0
                else imagined_sequences
            )

            self.actor_critic.learn(seq)

        # | ----------------------------------------------------- |#
        # | Step 5: Save the losses                               |#
        # | ----------------------------------------------------- |#
        self.losses = {
            key: np.mean(value) for key, value in self.world_model.loss.items()
        } | {
            "actor_loss": np.mean(self.actor_critic.actor_loss),
            "critic_loss": np.mean(self.actor_critic.critic_loss),
        }
        # raise valueerror if not dict[str,float]
        if not all(isinstance(v, float) for v in self.losses.values()):
            raise ValueError(
                "Losses should be a dictionary of string keys and float values."
            )
        # reset the actor and critic losses
        self.actor_critic.reset_losses()
        self.world_model.reset_loss()

        return self.losses

    def create_training_batch(self, real: int, imagined_sequences: list) -> list:
        """
        Creates a training batch for the Dreamer algorithm.

        Args:
            real (int): Percentage of real data to use in the batch (0-100).
            imagined_sequences (list): List of imagined sequences.

        Returns:
            list: List of training batches with real and imagined data.
        """
        if not (0 <= real <= 100):
            raise ValueError("Real percentage must be between 0 and 100.")

        n = int(real / 100 * self.batch_size)
        if n == 0:
            return imagined_sequences

        # Select n real samples from the replay buffer
        real_samples = self.real_RB.sample(n, self.horizon)

        # Combine real and imagined sequences
        return real_samples + imagined_sequences[: self.batch_size - n]
