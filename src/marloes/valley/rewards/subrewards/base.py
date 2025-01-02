from abc import ABC, abstractmethod

import numpy as np

from marloes.results.extractor import Extractor


class SubReward(ABC):
    """
    Represents an individual sub-reward with activation and scaling properties.
    """

    def __init__(self, active: bool = False, scaling_factor: float = 1.0):
        """
        Initializes the SubReward instance with activation and scaling properties.
        """
        self.active = active
        self.scaling_factor = scaling_factor

    @abstractmethod
    def calculate(
        self, extractor: Extractor, actual: bool, **kwargs
    ) -> float | np.ndarray:
        """
        Calculate the sub-reward based on the given extractor.
        """
        pass

    @staticmethod
    def _get_target(array: np.ndarray, i: int, actual: bool) -> float | np.ndarray:
        return array[i] if actual else array
