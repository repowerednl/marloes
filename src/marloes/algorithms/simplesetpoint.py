import random

from .base import BaseAlgorithm


class SimpleSetpoint(BaseAlgorithm):
    """
    Simple setpoint algorithm that sets the power of any asset to a random value between -1 and 1.
    """

    __name__ = "SimpleSetpoint"

    def __init__(self, config: dict, evaluate: bool = False):
        """
        Initializes the SimpleSetpoint algorithm.
        """
        super().__init__(config, evaluate)

    def get_actions(self, observations, deterministic: bool = False) -> dict:
        """
        Generates random actions for each agent in the environment.
        """
        return self.sample_actions(self.environment.agent_dict)

    def perform_training_steps(self, step: int) -> None:
        """
        Overrides the training step. No learning is required for the priority-based algorithm.
        """
        pass
