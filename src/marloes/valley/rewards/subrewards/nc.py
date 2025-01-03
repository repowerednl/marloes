import numpy as np

from marloes.results.extractor import Extractor
from marloes.valley.rewards.subrewards.base import SubReward


class NCSubReward(SubReward):
    """
    Sub-reward for penalizing electricity given to the grid (net congestion).
    """

    name = "NC"

    def calculate(
        self, extractor: Extractor, actual: bool, **kwargs
    ) -> float | np.ndarray:
        grid_state = self._get_target(extractor.grid_state, extractor.i, actual)
        return np.minimum(0, grid_state)
