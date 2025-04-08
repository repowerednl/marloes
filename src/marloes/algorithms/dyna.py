import logging

from marloes.algorithms.base_v2 import BaseAlgorithmV2


class Dyna(BaseAlgorithmV2):
    """
    Dyna algorithm that combines model-based and model-free reinforcement learning.
    """

    __name__ = "Dyna"

    def __init__(self, config: dict):
        """
        Initializes the Dyna algorithm.
        """
        super().__init__(config)
        self.world_model = None  # Placeholder for the world model
        self.SAC = None  # Placeholder for the SAC agent

        # Dyna specific parameters
        self.model_update_frequency = self.config.get("model_update_frequency", 100)
        self.k = self.config.get("k", 10)  # Model rollout horizon; planning steps
        self.model_updates_per_step = self.config.get("model_updates_per_step", 10)

    def get_actions(self, state: dict) -> dict:
        """
        Generates actions based on the current observation using the SAC agent.
        """
        # Convert state to tensor
        # TODO: Implement state conversion to tensor if needed

        # Get actions from the SAC agent
        actions = self.SAC.get_actions(state)
        return actions

    def perform_training_steps(self, step: int) -> None:
        """
        Placeholder for a single training step. To be overridden if needed.
        """
        # 1. Update world model (with real experiences only)
        # --------------------
        if step % self.model_update_frequency == 0 and step != 0:
            # Sample from real experiences
            real_batch = self.real_RB.sample(self.batch_size)

            # Update the world model with this batch
            self.world_model.update(real_batch)

        # 2. Generate synthetic experiences with the world model
        # --------------------
        # Get starting points for synthetic rollouts
        synthetic_states = self.real_RB.sample(self.batch_size)

        for _ in range(self.k):
            # Generate synthetic actions TODO: decide if random or policy
            synthetic_actions = self.environment.sample_actions()

            # Use the world model to predict next state and reward
            synthetic_next_state, synthetic_rewards = self.world_model.predict(
                synthetic_states, synthetic_actions
            )

            # Store synthetic experiences
            # TODO: keep in mind that this will be multiple states, so might need something different
            self.model_RB.push(
                synthetic_states,
                synthetic_actions,
                synthetic_rewards,
                synthetic_next_state,
            )
            synthetic_states = synthetic_next_state

        # 3. Update the model (SAC) with real and synthetic experiences
        # --------------------
        for _ in range(self.model_updates_per_step):
            # Sample from both real and synthetic experiences
            real_batch = self.real_RB.sample(self.batch_size)
            synthetic_batch = self.model_RB.sample(self.batch_size)

            # Combine batches
            combined_batch = self._combine_batches(real_batch, synthetic_batch)

            # Update the model (SAC) with the combined batch
            self.SAC.update(combined_batch)

    @staticmethod
    def _combine_batches(real_batch, synthetic_batch):
        """
        Combines real and synthetic batches for training.
        """
        pass
