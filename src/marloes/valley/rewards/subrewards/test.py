import numpy as np

from marloes.results.extractor import Extractor
from marloes.valley.rewards.subrewards.base import SubReward


class TESTSubReward(SubReward):
    """
    Test reward for testing purposes.
    - first we penalize solar production
    """

    name = "NB"

    def __init__(
        self,
        config: dict,
        active: bool = False,
        scaling_factor: float = 1.0,
    ):
        super().__init__(config, active, scaling_factor)

        self.max_production = sum(
            agent.get("AC", 1)  # max_power_out
            for agent in self.config["agents"]
            if agent.get("type") == "solar"
        )

    def calculate(
        self, extractor: Extractor, actual: bool, **kwargs
    ) -> float | np.ndarray:
        solar_production = self._get_target(
            extractor.total_solar_production, extractor.i, True
        )
        return -solar_production / self.max_production
