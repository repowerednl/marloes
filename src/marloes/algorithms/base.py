from abc import ABC, abstractmethod
from marloes.algorithms.types import AlgorithmType, parse_algorithm_type
from marloes.results.saver import Saver
from marloes.valley.env import EnergyValley


class BaseAlgorithm(ABC):
    """
    Abstract base class for energy optimization algorithms.
    """

    def __init__(self, config: dict):
        """
        Initializes the algorithm with a configuration dictionary.
        """
        self.saver = Saver()
        self.chunk_size = config.get("chunk_size", 0)
        self.algorithm_type = parse_algorithm_type(config.get("algorithm"))
        self.epochs = config.get("epochs", 100)
        self.environment = EnergyValley(config)

    def train(self) -> None:
        """
        Executes the training process for the algorithm.

        This method can be overridden by subclasses for algorithm-specific behavior.
        """
        observations, infos = self.environment.reset()

        for epoch in range(self.epochs):
            actions = self.get_actions(observations)
            observations, rewards, dones, infos = self.environment.step(actions)

            # Placeholder for training logic specific to subclasses
            self._train_step(observations, rewards, dones, infos)

            # After chunk is "full", it should be saved
            if epoch % self.chunk_size == 0:
                pass
                # self.saver.save()

        # self.saver.save_model()

    @abstractmethod
    def get_actions(self, observations) -> dict:
        """
        Generates actions based on the current observation.

        Returns:
            dict: Actions to take in the environment.
        """
        pass

    @abstractmethod
    def _train_step(self, observations, rewards, dones, infos) -> None:
        """
        Placeholder for a single training step. To be overridden if needed.
        """
        pass

    def load(self, uid: str) -> None:
        """
        Loads a parameter configuration from a file.
        """
        pass
