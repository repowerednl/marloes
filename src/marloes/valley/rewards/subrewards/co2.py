import numpy as np

from marloes.results.extractor import Extractor
from marloes.valley.rewards.subrewards.base import SubReward

MAX_DEMAND = 30
BATTERY_SCALE = 0.2


class CO2SubReward(SubReward):
    """
    Sub-reward for penalizing CO2 emissions.
    """

    name = "CO2"

    EMISSION_COEFFICIENTS = {
        "solar": 0.0,  # 45.5
        "wind": 0.0,  # 15.5
        "battery": 0.0,  # 70.0
        "electrolyser": 0.0,  # PEMWE: 15.25
        "grid": 284.73,
    }

    def __init__(
        self,
        config: dict,
        active: bool = False,
        scaling_factor: float = 1.0,
    ):
        """
        Initializes the CO2SubReward instance with activation and scaling properties.
        Stores max emissions used for normalization.
        """
        super().__init__(config, active, scaling_factor)
        max_demand = sum(
            agent.get("scale", 1) * MAX_DEMAND
            for agent in self.config["agents"]
            if agent.get("type") == "demand"
        )
        max_battery = sum(
            agent.get("power") * BATTERY_SCALE
            for agent in self.config["agents"]
            if agent.get("type") == "battery"
        )
        self.max_emmissions = (max_demand + max_battery) * self.EMISSION_COEFFICIENTS[
            "grid"
        ]  # Use the maximum emissions from the grid as the normalization factor

    def calculate(
        self, extractor: Extractor, actual: bool, **kwargs
    ) -> float | np.ndarray:
        i = extractor.i
        penalties = [
            self.EMISSION_COEFFICIENTS["solar"]
            * self._get_target(extractor.total_solar_production, i, actual),
            self.EMISSION_COEFFICIENTS["wind"]
            * self._get_target(extractor.total_wind_production, i, actual),
            self.EMISSION_COEFFICIENTS["battery"]
            * self._get_target(extractor.total_battery_production, i, actual),
            self.EMISSION_COEFFICIENTS["grid"]
            * self._get_target(extractor.total_grid_production, i, actual),
        ]
        return max(-(sum(penalties) / self.max_emmissions), -1)
