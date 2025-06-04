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
        grid_production = self._get_target(
            extractor.total_grid_production, extractor.i, True
        )
        battery_production = self._get_target(
            extractor.total_battery_production, extractor.i, True
        )

        demand = self._get_target(extractor.total_demand, extractor.i, True)

        solar_reward = solar_production
        grid_reward = -grid_production

        # Battery setpoint penalty
        battery_action = sum(
            extractor.__dict__[attr][extractor.i]
            for attr in extractor.__dict__
            if isinstance(getattr(extractor, attr), np.ndarray) and "Battery" in attr
        )

        surplus = (solar_production + grid_production + battery_production) - demand
        battery_setpoint_reward = -abs(battery_action - surplus)

        # print("Solar Reward:", solar_reward)
        # print("Grid Reward:", grid_reward)
        # print("Battery Setpoint Reward:", battery_setpoint_reward)

        return solar_reward / 2 + grid_reward + battery_setpoint_reward
