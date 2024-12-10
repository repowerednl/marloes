from abc import ABC, abstractmethod
from enum import Enum, auto
from marloes.valley.env import EnergyValley


class AlgorithmType(Enum):
    MODEL_BASED = auto()
    MODEL_FREE = auto()
    SOLVER = auto()
    SAC = auto()


class Algorithm(ABC):
    """
    Abstract base class for the algorithms
    """

    def __init__(self, config: dict, save_energy_flows: bool = False):
        self.set_algorithm(config.pop("algorithm"))
        self.epochs = config.pop("epochs", 100)  # 525600)  # 1 year in minutes
        self.valley = EnergyValley(config)
        self.saving = save_energy_flows
        # TODO: initialize the classes to save the energy flows
        self.flows = [] if self.saving else None

    def set_algorithm(self, alg: str):
        if alg == "model_based":
            self.algorithm = AlgorithmType.MODEL_BASED
        elif alg == "model_free":
            self.algorithm = AlgorithmType.MODEL_FREE
        elif alg == "solver":
            self.algorithm = AlgorithmType.SOLVER
        elif alg == "sac":
            self.algorithm = AlgorithmType.SAC
        else:
            raise ValueError(f"Unknown algorithm type: {alg}")

    @abstractmethod
    def get_actions(self, observation):
        pass

    @abstractmethod
    def train(self, observation, reward, done, info):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass
