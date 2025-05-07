import numpy as np

from marloes.results.extractor import Extractor
from marloes.valley.rewards.subrewards.base import SubReward


class SSSubReward(SubReward):
    """
    Sub-reward for penalizing lack of self-sufficiency.
    """

    name = "SS"

    def calculate(
        self,
        extractor: Extractor,
        actual: bool,
        net_grid_state: float,
        net_demand: float,
        net_battery_intake: float,
        **kwargs
    ) -> float | np.ndarray:
        if actual:
            penalty = net_grid_state
            normalized_penalty = penalty / (net_demand + net_battery_intake)
            return -normalized_penalty
        return -np.maximum(0, np.cumsum(extractor.grid_state))
