from abc import ABC, abstractmethod
from enum import Enum


class AlgorithmType(Enum):
    MODEL_BASED = "model_based"
    MODEL_FREE = "model_free"
    SOLVER = "solver"


class Algorithm(ABC):
    """
    Abstract base class for the algorithms
    """

    @abstractmethod
    def __init__(self, config: dict):
        pass

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
