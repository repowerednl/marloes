import numpy as np

from marloes.results.extractor import Extractor
from marloes.valley.rewards.subrewards.base import SubReward


class NBSubReward(SubReward):
    """
    Sub-reward for incentivizing net-positive energy balance.
    """

    name = "NB"

    def calculate(
        self, extractor: Extractor, actual: bool, prev_net_grid_state: float, **kwargs
    ) -> float | np.ndarray:
        if actual:
            return -prev_net_grid_state
        return -np.cumsum(extractor.grid_state)
