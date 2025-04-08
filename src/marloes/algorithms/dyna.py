import logging

from marloes.algorithms.base import BaseAlgorithm
from marloes.data.replaybuffer import ReplayBuffer


N1 = 1e5  # Sizeable; can be configured
N2 = 1e4  # Smaller; since synthetic experiences can become stale?
num_initial_random_steps = 1000  # Configurable
model_update_frequency = 100
batch_size = 128
k = 10  # Model rollout horizon; planning steps
model_updates_per_step = 10  # Number of model updates per step


class Dyna(BaseAlgorithm):
    """
    Dyna algorithm that combines model-based and model-free reinforcement learning.
    """

    __name__ = "Dyna"

    def __init__(self, config: dict):
        """
        Initializes the Dyna algorithm.
        """
        super().__init__(config)
        self.real_RB = ReplayBuffer(capacity=N1, device=self.device)
        self.model_RB = ReplayBuffer(capacity=N2, device=self.device)
        self.world_model = None  # Placeholder for the world model
        self.SAC = None  # Placeholder for the SAC agent

    def train(self) -> None:
        # Initialization
        state, infos = self.environment.reset()

        # Training loop
        for step in range(self.epochs):
            # 1. Collect data from environment
            # --------------------
            if step < num_initial_random_steps:
                # Initially do random actions for exploration
                actions = self.environment.sample_actions()
            else:
                # Use policy (will be SAC actor)
                actions = self.get_actions(state)

            next_state, rewards, dones, infos = self.environment.step(actions)

            # Store real experiences
            self.real_RB.push(state, actions, rewards, next_state)

            state = next_state

            # 2. Update world model (with real experiences only)
            # --------------------
            if step % model_update_frequency == 0 and step != 0:
                # Sample from real experiences
                real_batch = self.real_RB.sample(batch_size)

                # Update the world model with this batch
                self.world_model.update(real_batch)

            # 3. Generate synthetic experiences with the world model
            # --------------------
            # Get starting points for synthetic rollouts
            synthetic_states = self.real_RB.sample(batch_size)

            for _ in range(k):
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

            # 4. Update the model (SAC) with real and synthetic experiences
            # --------------------
            for _ in range(model_updates_per_step):
                # Sample from both real and synthetic experiences
                real_batch = self.real_RB.sample(batch_size)
                synthetic_batch = self.model_RB.sample(batch_size)

                # Combine batches
                combined_batch = self._combine_batches(real_batch, synthetic_batch)

                # Update the model (SAC) with the combined batch
                self.SAC.update(combined_batch)
