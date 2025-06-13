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
        "solar": 45.5,
        "wind": 15.5,
        "battery": 70.0,
        "electrolyser": 15.25,  # PEMWE
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
            handler.get("scale", 1) * MAX_DEMAND
            for handler in self.config["handlers"]
            if handler.get("type") == "demand"
        )
        max_battery = sum(
            handler.get("power") * BATTERY_SCALE
            for handler in self.config["handlers"]
            if handler.get("type") == "battery"
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
