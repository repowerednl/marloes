import numpy as np

from marloes.results.extractor import Extractor
from marloes.valley.rewards.subrewards.base import SubReward


class SSSubReward(SubReward):
    """
    Sub-reward for penalizing lack of self-sufficiency.
    """

    def calculate(
        self, extractor: Extractor, actual: bool, prev_net_grid_state: float, **kwargs
    ) -> float | np.ndarray:
        if actual:
            return -max(0, prev_net_grid_state)
        return -np.maximum(0, np.cumsum(extractor.grid_state))
