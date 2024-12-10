from abc import ABC, abstractmethod
import random
from enum import Enum, auto
from marloes.valley.env import EnergyValley


class AlgorithmType(Enum):
    MODEL_BASED = auto()
    MODEL_FREE = auto()
    SOLVER = auto()
    MADDPG = auto()


class Algorithm(ABC):
    """
    Abstract base class for the algorithms
    """

    def __init__(self, config: dict, save_energy_flows: bool = False):
        self._set_algorithm(config.pop("algorithm"))
        self.epochs = config.pop("epochs", 100)  # 525600)  # 1 year in minutes
        self.valley = EnergyValley(config)
        self.saving = save_energy_flows
        # TODO: initialize the classes to save the energy flows
        self.flows = [] if self.saving else None

    def _set_algorithm(self, alg: str):
        if alg == "model_based":
            self.algorithm = AlgorithmType.MODEL_BASED
        elif alg == "model_free":
            self.algorithm = AlgorithmType.MODEL_FREE
        elif alg == "solver":
            self.algorithm = AlgorithmType.SOLVER
        elif alg == "maddpg":
            self.algorithm = AlgorithmType.MADDPG
        else:
            raise ValueError(f"Unknown algorithm type: {alg}")

    def train(self):
        """
        Run the simulation/training phase of the algorithm, can be overridden by subclasses.
        """
        # Get the initial observation
        observation = self.valley.reset()

        for epoch in range(self.epochs):
            # Create an actions dictionary using the agent IDs
            actions = {agent.id: random.random() for agent in self.valley.agents}

            # Take a step in the environment
            observation, reward, done, info = self.valley.step(actions)

            # Save the energy flows if saving is enabled
            if self.saving:
                self.flows.append(info)

            # TODO: Train the algorithm
            # TODO: Save the training results

        # Stash any final results

    @abstractmethod
    def get_actions(self, observation):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass
