import logging
from abc import ABC, abstractmethod

from marloes.results.saver import Saver
from marloes.valley.env import EnergyValley
from marloes.networks.util import obs_to_tens


class BaseAlgorithm(ABC):
    """
    Abstract base class for energy optimization algorithms.
    """

    # Registry for all subclasses of BaseAlgorithm
    _registry = {}

    def __init__(self, config: dict):
        """
        Initializes the algorithm with a configuration dictionary.
        """
        logging.info(
            f"Initializing {self.__class__.__name__} algorithm and setting up the environment..."
        )
        self.saver = Saver(config=config)
        self.chunk_size = config.get("chunk_size", 10000)
        self.epochs = config.get("epochs", 100000)
        self.environment = EnergyValley(config, self.__class__.__name__)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        BaseAlgorithm._registry[cls.__name__] = cls

    def train(self) -> None:
        """
        Executes the training process for the algorithm.

        This method can be overridden by subclasses for algorithm-specific behavior.
        """
        logging.info("Starting training process...")
        observations, infos = self.environment.reset()

        for epoch in range(self.epochs):
            if epoch % 1000 == 0:
                logging.info(f"Reached epoch {epoch}/{self.epochs}...")

            # Get actions
            actions = self.get_actions(observations)
            observations, rewards, dones, infos = self.environment.step(actions)
            # Add to ReplayBuffer TODO: Implement ReplayBuffer MAR-141
            # ReplayBuffer only wants tensors, so we need to convert the observations

            # For x timesteps, perform the training step on a sample from the ReplayBuffer
            self._train_step(observations, rewards, dones, infos)

            # After chunk is "full", it should be saved
            if self.chunk_size != 0 and epoch % self.chunk_size == 0 and epoch != 0:
                logging.info("Saving intermediate results and resetting extractor...")
                self.saver.save(extractor=self.environment.extractor)
                # clear the extractor
                self.environment.extractor.clear()

        # Save the final results and TODO: model
        logging.info("Training finished. Saving results...")
        self.saver.final_save(self.environment.extractor)

        logging.info("Training process completed.")

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
        TODO: Implement loading of model parameters.
        """
        pass

    @staticmethod
    def get_algorithm(name: str, config: dict):
        """
        Retrieve the correct subclass based on its name.
        """
        if name not in BaseAlgorithm._registry:
            raise ValueError(
                f"Algorithm '{name}' is not registered as a subclass of BaseAlgorithm."
            )
        return BaseAlgorithm._registry[name](config)

    @staticmethod
    def _get_net_forecasted_power(observations: dict, period: int = 60) -> float:
        """
        Looks at the forecasts of each supply and demand agent to calculate the net power.
        Sum the forecasts of the next period, period is in minutes, defaults to 60 min (1 hour).
        """
        forecasts = [
            observations[agent]["forecast"]
            for agent in observations.keys()
            if "forecast" in observations[agent]
        ]
        if forecasts:
            # Ensure the period does not exceed the length of any forecast
            period = min(period, min(len(forecast) for forecast in forecasts))
        else:
            # if no forecasts are available, return 0.0
            return 0.0

        return sum(sum(forecast[:period]) for forecast in forecasts)
