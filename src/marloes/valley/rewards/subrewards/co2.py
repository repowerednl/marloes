import numpy as np

from marloes.results.extractor import Extractor
from marloes.valley.rewards.subrewards.base import SubReward


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
        return -sum(penalties)
