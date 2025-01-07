from .base import BaseAlgorithm
import random


class SimpleSetpoint(BaseAlgorithm):
    """
    Simple setpoint algorithm that sets the power of any asset to a random value between -1 and 1.
    """

    __name__ = "SimpleSetpoint"

    def __init__(self, config: dict):
        """
        Initializes the SimpleSetpoint algorithm.
        """
        super().__init__(config)

    def get_actions(self, observations) -> dict:
        """
        Generates random actions for each agent in the environment.
        """
        return {agent_id: random.uniform(-1, 1) for agent_id in observations.keys()}

    def _train_step(self, observations, rewards, dones, infos) -> None:
        """
        Overrides the training step. No learning is required for the SimpleSetpoint algorithm.
        """
        pass
