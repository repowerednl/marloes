import logging
from abc import ABC, abstractmethod
import torch

from marloes.results.saver import Saver
from marloes.valley.env import EnergyValley
from marloes.networks.util import dict_to_tens
from marloes.data.replaybuffer import ReplayBuffer


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
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # for future ticket, make sure this can run on GPU instead of CPU

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        BaseAlgorithm._registry[cls.__name__] = cls

    def train(self, update_step: int = 100) -> None:
        """
        Executes the training process for the algorithm.

        This method can be overridden by subclasses for algorithm-specific behavior.
        """
        logging.info("Starting training process...")
        observations, infos = self.environment.reset()
        capacity = 1000 * update_step
        logging.info(f"Initializing ReplayBuffer with capacity {capacity}...")
        RB = ReplayBuffer(capacity=capacity, device=self.device)

        for epoch in range(self.epochs):
            if epoch % 1000 == 0:
                logging.info(f"Reached epoch {epoch}/{self.epochs}...")

            # Gathering experiences
            obs = dict_to_tens(observations, concatenate_all=True)
            actions = self.get_actions(observations)
            observations, rewards, dones, infos = self.environment.step(actions)
            # Add to ReplayBuffer
            acts = dict_to_tens(actions, concatenate_all=True)
            rew = dict_to_tens(rewards, concatenate_all=True)
            rew = torch.tensor([rew.sum()])
            RB.push(obs, acts, rew)

            # For x timesteps, perform the training step on a sample (update_step size) from the ReplayBuffer
            if epoch % update_step == 0 and epoch != 0:
                logging.info("Performing training step...")
                sample = RB.sample(update_step, True)
                dones = torch.ones(update_step)
                # passing artificial dones: a tensor with all continuation (1) flags, except for the last one (0) - length x
                dones[-1] = 0
                self._train_step(
                    sample["obs"],
                    sample["action"],
                    sample["reward"],
                )

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
