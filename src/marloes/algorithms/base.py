import logging
from abc import ABC, abstractmethod
import random
import torch

from marloes.results.saver import Saver
from marloes.valley.env import EnergyValley
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
        self.config = config

        # Initialize the Saver, environment, and device
        self.saver = Saver(config=config)
        self.environment = EnergyValley(config, self.__class__.__name__)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # for future ticket, make sure this can run on GPU instead of CPU

        # General settings
        self.chunk_size = self.config.get("chunk_size", 10000)
        self.training_steps = self.config.get("training_steps", 100000)
        self.num_initial_random_steps = self.config.get("num_initial_random_steps", 0)
        self.batch_size = self.config.get("batch_size", 128)

        # Initialize ReplayBuffers
        try:
            self.real_RB = ReplayBuffer(
                capacity=self.config["replay_buffers"].get("real_capacity", 1000),
                device=self.device,
            )
        except KeyError:
            self.real_RB = ReplayBuffer(
                capacity=10000,  # Default capacity if not specified
                device=self.device,
            )
        try:
            self.model_RB = ReplayBuffer(
                capacity=self.config["replay_buffers"].get("model_capacity", 1000),
                device=self.device,
            )
        except KeyError:
            self.model_RB = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        BaseAlgorithm._registry[cls.__name__] = cls

    def train(self) -> None:
        """
        Executes the training process for the algorithm.

        This method can be overridden by subclasses for algorithm-specific behavior.
        """
        # Initialization
        logging.info("Starting training process...")
        state, infos = self.environment.reset()

        # Main training loop
        for step in range(self.training_steps):
            if step % 1000 == 0:
                logging.info(f"Reached step {step}/{self.training_steps}...")

            # 1. Collect data from environment
            # --------------------
            if step < self.num_initial_random_steps:
                # Initially do random actions for exploration
                actions = self.sample_actions(self.environment.agent_dict)
            else:
                # Get actions from the algorithm
                actions = self.get_actions(state)

            next_state, rewards, dones, infos = self.environment.step(actions)

            # Store (real) experiences
            self.real_RB.push(state, actions, rewards, next_state)

            state = next_state

            # 2. Perform algorithm-specific training steps
            # --------------------
            self.perform_training_steps(step)

            # Any time a chunk is "full", it should be saved
            if self.chunk_size != 0 and step % self.chunk_size == 0 and step != 0:
                logging.info("Saving intermediate results and resetting extractor...")
                self.saver.save(extractor=self.environment.extractor)
                # clear the extractor
                self.environment.extractor.clear()

        # Save the final results and TODO: model
        logging.info("Training finished. Saving results...")
        self.saver.final_save(self.environment.extractor)

        logging.info("Training process completed.")

    @abstractmethod
    def get_actions(self, state) -> dict:
        """
        Generates actions based on the current observation.

        Returns:
            dict: Actions to take in the environment.
        """
        pass

    @abstractmethod
    def perform_training_steps(self, step: int) -> None:
        """
        Placeholder for a single training step. To be overridden.
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

    def sample_actions(self, agent_dict: dict) -> dict:
        """
        Generates random actions for each agent in the environment.
        """
        return {agent_id: random.uniform(-1.0, 1.0) for agent_id in agent_dict.keys()}

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
