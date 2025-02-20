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
        """
        i = extractor.i
        penalties = [
            abs(
                self._get_target(extractor.total_solar_production, i, actual)
                - self._get_target(extractor.total_solar_nomination, i, actual)
            ),
            abs(
                self._get_target(extractor.total_wind_production, i, actual)
                - self._get_target(extractor.total_wind_nomination, i, actual)
            ),
        ]
        return -sum(penalties)
