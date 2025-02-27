import numpy as np

from marloes.results.extractor import Extractor
from marloes.valley.rewards.subrewards.base import SubReward


class NESubReward(SubReward):
    """
    NE: Nomination Error
    Sub-reward that incentivizes following nomination. Any deviation from the nomination is penalized.
    """

    name = "NE"

    def calculate(
        self, extractor: Extractor, actual: bool, **kwargs
    ) -> float | np.ndarray:
        """
        Calculates the difference between the power and nomination of all supply (wind and solar) assets.
        Is calculated every hour with the difference between the mean of the total production and the nomination.

        """
        # i is current timestep (step-size is 1 minute), starting at 00:00:00 at i=0, therefore i % 60 == 0 is true every hour
        if extractor.i % 60 != 0 and extractor.i != 0:
            return 0
        last_hour = slice(extractor.i - 60, extractor.i)
        # get total solar information: mean production over the last hour, nomination
        solar_production = self._get_target(
            extractor.total_solar_production, last_hour, actual
        )
        solar_nomination = self._get_target(
            extractor.total_solar_nomination, last_hour, actual
        )
        # get total wind information: production, nomination
        wind_production = self._get_target(
            extractor.total_wind_production, last_hour, actual
        )
        wind_nomination = self._get_target(
            extractor.total_wind_nomination, last_hour, actual
        )
        # calculate the penalty for solar and wind
        penalties = [
            abs(solar_production - solar_nomination),
            abs(wind_production - wind_nomination),
        ]
        return -sum(penalties)
