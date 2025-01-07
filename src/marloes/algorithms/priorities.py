from .base import BaseAlgorithm


class Priorities(BaseAlgorithm):
    """
    Priority-based algorithm that solves energy flows using pre-defined priorities.
    """

    __name__ = "Priorities"

    def __init__(self, config: dict):
        """
        Initializes the Priorities algorithm.
        """
        super().__init__(config)

    def get_actions(self, observations) -> dict:
        """
        No explicit actions are needed as the priorities are predefined.
        """
        return {}

    def _train_step(self, observations, rewards, dones, infos) -> None:
        """
        Overrides the training step. No learning is required for the priority-based algorithm.
        """
        pass
