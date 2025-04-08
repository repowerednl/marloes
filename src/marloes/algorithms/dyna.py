import torch

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
        self.real_sample_ratio = self.config.get("real_sample_ratio", 0.5)

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
        Performs the training steps for the Dyna algorithm, containing the following:
        1. Update the world model with real experiences.
        2. Generate synthetic experiences using the world model.
        3. Update the model (SAC) with both real and synthetic experiences.
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
            synthetic_actions = self.sample_actions(self.environment.agent_dict)

            # Use the world model to predict next state and reward
            synthetic_next_states, synthetic_rewards = self.world_model.predict(
                synthetic_states, synthetic_actions
            )

            # Store synthetic experiences, one by one
            for i in range(self.batch_size):
                _state = {key: val[i] for key, val in synthetic_states.items()}
                _actions = {key: val[i] for key, val in synthetic_actions.items()}
                _rewards = {key: val[i] for key, val in synthetic_rewards.items()}
                _next_state = {
                    key: val[i] for key, val in synthetic_next_states.items()
                }
                self.model_RB.push(_state, _actions, _rewards, _next_state)

            synthetic_states = synthetic_next_states

        # 3. Update the model (SAC) with real and synthetic experiences
        # --------------------
        for _ in range(self.model_updates_per_step):
            # Sample from both real and synthetic experiences
            real_batch = self.real_RB.sample(self.batch_size * self.real_sample_ratio)
            synthetic_batch = self.model_RB.sample(
                self.batch_size * (1 - self.real_sample_ratio)
            )

            # Combine batches
            combined_batch = self._combine_batches(real_batch, synthetic_batch)

            # Update the model (SAC) with the combined batch
            self.SAC.update(combined_batch)

    @staticmethod
    def _combine_batches(real_batch, synthetic_batch):
        """
        Combines real and synthetic batches for training.
        """
        combined_batch = {}
        for key in real_batch.keys():
            # Concatenate the corresponding tensors along the 0th dimension.
            combined_batch[key] = torch.cat(
                [real_batch[key], synthetic_batch[key]], dim=0
            )
        return combined_batch
