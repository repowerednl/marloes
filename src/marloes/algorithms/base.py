from abc import ABC, abstractmethod
from enum import Enum, auto
from marloes.results.saver import Saver


class AlgorithmType(Enum):
    MODEL_BASED = auto()
    MODEL_FREE = auto()
    SOLVER = auto()


class Algorithm(ABC):
    """
    Abstract base class for the algorithms
    """

    @abstractmethod
    def __init__(self, config: dict):
        self.saver = Saver(config)
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
