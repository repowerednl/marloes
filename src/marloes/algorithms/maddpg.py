from .base import BaseAlgorithm


# TODO: inherit from the preconfigured algorithm
class MADDPG(BaseAlgorithm):
    def __init__(self, config: dict):
        """
        Initializes the MADDPG.
        """
        super().__init__(config)

    def train(self) -> None:
        pass
