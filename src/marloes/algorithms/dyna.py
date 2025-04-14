import torch

from marloes.algorithms import BaseAlgorithm, SAC
from marloes.networks.simple_world_model.world_model import WorldModel


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
        self.world_model = WorldModel(self.config)
        self.sac = SAC(self.config)

        # Dyna specific parameters
        dyna_config = config.get("dyna", {})
        self.model_update_frequency = dyna_config.get("model_update_frequency", 100)
        self.k = dyna_config.get("k", 10)  # Model rollout horizon; planning steps
        self.model_updates_per_step = dyna_config.get("model_updates_per_step", 10)
        self.real_sample_ratio = dyna_config.get("real_sample_ratio", 0.5)

    def get_actions(self, state: dict) -> dict:
        """
        Generates actions based on the current observation using the SAC agent.
        """
        # Convert state to tensor
        state_tensor = self.real_RB._convert_to_tensors([state])

        # Get actions from the SAC agent
        actions = self.sac.act(state_tensor)

        # Convert actions back to the original format
        action_list = actions.squeeze(0).tolist()
        action_dict = {
            key: action_list[i]
            for i, key in enumerate(self.environment.agent_dict.keys())
        }

        return action_dict

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
            real_batch = self.real_RB.sample(self.batch_size, flatten=False)

            # Update the world model with this batch
            self.world_model.update(real_batch, self.device)

        # 2. Generate synthetic experiences with the world model
        # --------------------
        # Get starting points for synthetic rollouts
        sample = self.real_RB.sample(self.batch_size, flatten=False)
        synthetic_states = [transition.state for transition in sample]

        for _ in range(self.k):
            # Generate synthetic actions TODO: decide if random or policy
            synthetic_actions = [
                self.sample_actions(self.environment.agent_dict)
                for _ in range(self.batch_size)
            ]

            # Use the world model to predict next state and reward
            synthetic_next_states, synthetic_rewards = self.world_model.predict(
                synthetic_states, synthetic_actions, device=self.device
            )

            # Store synthetic experiences
            for i in range(self.batch_size):
                self.model_RB.push(
                    synthetic_states[i],
                    synthetic_actions[i],
                    synthetic_rewards[i],
                    synthetic_next_states[i],
                )

            synthetic_states = synthetic_next_states

        # 3. Update the model (SAC) with real and synthetic experiences
        # --------------------
        for _ in range(self.model_updates_per_step):
            # Sample from both real and synthetic experiences; SAC uses flattened batches
            real_batch = self.real_RB.sample(
                int(self.batch_size * self.real_sample_ratio), flatten=True
            )
            synthetic_batch = self.model_RB.sample(
                int(self.batch_size * (1 - self.real_sample_ratio)), flatten=True
            )

            # Combine batches
            combined_batch = self._combine_batches(real_batch, synthetic_batch)

            # Update the model (SAC) with the combined batch
            self.sac.update(combined_batch)

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
