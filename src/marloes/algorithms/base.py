import logging
from abc import ABC, abstractmethod
import random
import time
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

        # Initialize the Saver, environment, and device
        self.saver = Saver(config=config)
        self.environment = EnergyValley(config, self.__class__.__name__)

        # Update config with environment parameters
        config["num_agents"] = len(self.environment.agent_dict)
        config["state_dim"] = self.environment.state_dim[0]
        config["action_dim"] = self.environment.action_dim[0]
        config["global_dim"] = self.environment.global_dim[0]
        config["agents_scalar_dim"] = self.environment.agents_scalar_dim
        config["forecasts"] = self.environment.forecasts

        self.config = config
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.device = torch.device("cpu")
        if self.device.type == "cpu":
            logging.warning(
                "MPS is not available. Using CPU for computations. Performance may be slower."
            )

        # General settings
        self.chunk_size = config.get("chunk_size", 10000)
        self.training_steps = config.get("training_steps", 100000)
        num_initial_random_steps = config.get("num_initial_random_steps", 0)
        self.batch_size = config.get("batch_size", 128)
        self.num_initial_random_steps = max(
            num_initial_random_steps, self.batch_size
        )  # Ensure batch size is not larger than initial random steps

        # Initialize ReplayBuffers
        replay_buffer_config = config.get("replay_buffers", {})
        self.real_RB = ReplayBuffer(
            capacity=replay_buffer_config.get("real_capacity", 1000),
            device=self.device,
        )
        try:
            self.model_RB = ReplayBuffer(
                capacity=replay_buffer_config.get("model_capacity", 1000),
                device=self.device,
            )
        except KeyError:
            self.model_RB = None

        # Save losses
        self.losses = {}
        self.normalize = True

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
            if step % (self.training_steps // 100) == 0:
                logging.info(f"Reached step {step}/{self.training_steps}...")

            # 1. Collect data from environment
            # --------------------
            if step < self.num_initial_random_steps:
                # Initially do random actions for exploration
                actions = self.sample_actions(self.environment.agent_dict)
            else:
                # Get actions from the algorithm
                actions = self.get_actions(state)
            next_state, reward, dones, infos = self.environment.step(
                actions=actions,
                loss_dict=self.losses,
                normalize=self.normalize,
            )

            # Store (real) experiences
            self.real_RB.push(state, actions, reward, next_state)

            state = next_state

            # 2. Perform algorithm-specific training steps
            # --------------------
            if step > self.num_initial_random_steps:
                self.losses = self.perform_training_steps(step)
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
    def perform_training_steps(self, step: int) -> dict[str, float]:
        """
        Placeholder for a single training step. To be overridden.
        Should return a dict with the losses of the (parts of) the model.
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
