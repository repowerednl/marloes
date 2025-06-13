from datetime import datetime
import numpy as np

from marloes.results.extractor import Extractor
from marloes.valley.rewards.subrewards.base import SubReward

import logging

SOLAR_SCALE = 0.4


class NESubReward(SubReward):
    """
    NE: Nomination Error
    Sub-reward that incentivizes following nomination. Any deviation from the nomination is penalized.
    """

    name = "NE"

    def __init__(
        self,
        config: dict,
        active: bool = False,
        scaling_factor: float = 1.0,
    ):
        """
        Initializes the NESubReward instance with activation and scaling properties.
        """
        super().__init__(config, active, scaling_factor)

        max_solar_ac = sum(
            handler.get("AC") * SOLAR_SCALE
            for handler in self.config["handlers"]
            if handler.get("type") == "solar"
        )
        max_wind_ac = sum(
            handler.get("AC") * SOLAR_SCALE
            for handler in self.config["handlers"]
            if handler.get("type") == "wind"
        )
        self.max_nomination_divergence = max_solar_ac + max_wind_ac

    def calculate(
        self, extractor: Extractor, actual: bool, time_stamp: datetime, **kwargs
    ) -> float | np.ndarray:
        """
        Calculates the difference between the power and nomination of all supply (wind and solar) assets.
        Is calculated every hour with the difference between the mean of the total production and the nomination.
        """
        minutes = slice(extractor.i - time_stamp.minute, extractor.i)
        nom_sofar = self._get_total_nomination_sofar(extractor, minutes)
        prod_sofar = self._get_total_production_sofar(extractor, minutes)

        if actual:
            prod_cum = sum(prod_sofar) / 60  # kWh
            nom_cum = sum(nom_sofar) / 60  # kWh

            mismatch = abs(prod_cum - nom_cum)
            return max(-(mismatch / self.max_nomination_divergence), -1.0)
        else:
            raise NotImplementedError(
                "The NE subreward is not implemented for the non-actual case."
            )

    def _get_total_nomination_sofar(
        self, extractor: Extractor, minutes: slice
    ) -> float | np.ndarray:
        """
        Returns the total nomination so far.
        If actual:
            - Use the current timestep nominations
        If not actual:
            - Sum the full arrays of nominations
        """
        total_nomination = sum(
            self._get_target(nom, minutes, True)
            for nom in [
                extractor.total_solar_nomination,
                extractor.total_wind_nomination,
                extractor.total_demand_nomination,
            ]
        )
        return total_nomination

    def _get_total_production_sofar(
        self, extractor: Extractor, minutes: slice
    ) -> float | np.ndarray:
        """
        Returns the total production so far.
        If actual:
            - Use the current timestep nominations
        If not actual:
            - Sum the full arrays of nominations
        """
        total_production = sum(
            self._get_target(prod, minutes, True)
            for prod in [
                extractor.total_solar_production,
                extractor.total_wind_production,
            ]
        )
        # Demmand is positive, so we need to subtract it
        total_production -= extractor.total_demand[minutes]
        return total_production
