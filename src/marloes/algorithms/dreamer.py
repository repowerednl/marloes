from .base import BaseAlgorithm
from marloes.networks.dreamer.WorldModel import WorldModel
from marloes.networks.dreamer.ActorCritic import ActorCritic

import torch


class Dreamer(BaseAlgorithm):
    """
    Dreamer algorithm implementation using the WorldModel and ActorCritic modules.
    """

    def __init__(self, config: dict):
        """
        Initializes the Dreamer algorithm.

        Args:
            config (dict): Configuration dictionary for the algorithm.
        """
        super().__init__(config)
        self._initialize_world_model()
        self._initialize_actor_critic()
        self.update_interval = self.config.get("update_interval", 100)
        self.previous = None
        self.horizon = 16

    def _initialize_world_model(self):
        """
        Initializes the WorldModel with observation and action shapes.
        """
        self.world_model = WorldModel(
            state_dim=self.environment.state_dim,
            action_dim=self.environment.action_dim,
        )

    def _initialize_actor_critic(self):
        """
        Initializes the actor and critic networks.
        """
        input_size = (
            self.world_model.rssm.hidden_size + self.world_model.rssm.latent_state_size
        )
        self.actor_critic = ActorCritic(
            input=input_size,
            output=self.environment.action_dim[0],
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

    def get_actions(self, observations):
        """
        Computes actions based on the current observations and model state.

        Args:
            observations (torch.Tensor): Current observations.

        Returns:
            dict: Dictionary mapping agent IDs to actions.
        """
        if not self.previous:
            self._init_previous()
        # set world_model to eval mode
        # set actor_critic to eval mode
        with torch.no_grad():
            # Step 1: Get the recurrent state (based on previous state)  #
            # ---------------------------------------------------------- #
            h_t, _, _ = self.world_model.rssm.forward(
                self.previous["h_t"], self.previous["z_t"], self.previous["a_t"]
            )
            h_t = h_t[-1]
            h_t = h_t.view(h_t.size(0), -1)  # Flatten the hidden state
            # Step 2: Get the latent state (based on current obs and h_t)  #
            # ------------------------------------------------------------ #
            x = torch.cat([observations.unsqueeze(0), h_t], dim=-1)
            z_t, _ = self.world_model.rssm.encoder(x)

            # Step 3: Get the action (based on the model state)  #
            # -------------------------------------------------- #
            s = torch.cat([h_t, z_t], dim=-1)
            actions = self.actor_critic.act(s)

            # Step 4: Update the previous state with the current state  #
            # -------------------------------------------------- #
            self.previous["h_t"] = h_t
            self.previous["z_t"] = z_t
            self.previous["a_t"] = actions

            return {
                agent_id: actions[i]
                for i, agent_id in enumerate(self.environment.agent_dict.keys())
            }

    def perform_training_steps(self, step: int):
        """
        Executes a training step for the Dreamer algorithm.

        Args:
            step (int): Current training step.

        Returns:
            dict: Dictionary containing world model and actor-critic losses.
        """
        if step % self.update_interval != 0 and step > 0:
            return
        # set world_model to train mode
        # set actor_critic to train mode

        # | --------------------------------------------------- |#
        # | Step 1: Get a sample from the replay buffer         |#
        # |  - should be a sample of sequences (size=horizon)   |#
        # | --------------------------------------------------- |#
        real_sample = self.real_RB.sample(
            self.batch_size, self.horizon
        )  # sequence = size horizon

        # | ----------------------------------------------------- |#
        # | Step 2: Update the world model with real interactions |#
        # | ----------------------------------------------------- |#
        worldmodel_losses = self.world_model.learn(real_sample)

        # | ----------------------------------------------------- |#
        # | Step 3: Imagine trajectories for ActorCritic learning |#
        # |  - Sample starting point from the replay buffer       |#
        # |  - Pass Actor to the imagine function in WorldModel   |#
        # |  - Should return a batch of imagined sequences        |#
        # | ----------------------------------------------------- |#
        starting_points = self.real_RB.sample(self.batch_size)
        imagined_sequences = self.world_model.imagine(
            starting_points["state"], self.actor_critic.actor, self.horizon
        )

        # | ------------------------------------- |#
        # | Step 4: Update the actor-critic model |#
        # | ------------------------------------- |#
        # Only update with imagined trajectories
        # (updating with real trajectories should be implemented for:
        # - environments where the reward is tricky to predict)

        actorcritic_losses = self.actor_critic.learn(imagined_sequences)

        # | ----------------------------------------------------- |#
        # | Step 5: Save the losses                               |#
        # | ----------------------------------------------------- |#
        # to Extractor here?
        # returning losses to Base Algorithm might be cleaner
        return {
            "world": worldmodel_losses,
            "actorcritic": actorcritic_losses,
        }
