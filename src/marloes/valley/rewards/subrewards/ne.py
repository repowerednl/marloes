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
        if actual:
            return (
                -self._calculate_penalty(
                    extractor, slice(extractor.i - 60, extractor.i), actual=True
                )
                if self._hour_has_passed(extractor.i)
                else self._calculate_intermediate_penalty(extractor, actual=True)
            )

        # TODO: add intermediate penalty for existing data
        reward_array = np.zeros(len(extractor.total_solar_production))
        for t in range(60, len(reward_array), 60):
            reward_array[t] = -self._calculate_penalty(
                extractor, slice(t - 60, t), actual=False
            )

        return reward_array

    def _calculate_penalty(
        self, extractor: Extractor, time_slice: slice, actual: bool
    ) -> float:
        """
        The penalty calculations for the Nomination Error sub-reward.
        """
        solar_production = self._get_target(
            extractor.total_solar_production, time_slice, actual
        )
        solar_nomination = self._get_target(
            extractor.total_solar_nomination, time_slice, actual
        )
        wind_production = self._get_target(
            extractor.total_wind_production, time_slice, actual
        )
        wind_nomination = self._get_target(
            extractor.total_wind_nomination, time_slice, actual
        )
        # TODO: Add all production and all nomation together (NB: demand nomination is negative)
        solar_penalty = abs(np.mean(solar_production) - np.mean(solar_nomination))
        wind_penalty = abs(np.mean(wind_production) - np.mean(wind_nomination))

        return solar_penalty + wind_penalty

    @staticmethod
    def _hour_has_passed(i: int) -> bool:
        return i % 60 == 0 and i != 0

    def _calculate_intermediate_penalty(
        self, extractor: Extractor, actual: bool
    ) -> float:
        """
        Returns the difference between actual nomination_fraction and the expected nomination_fraction as a small penalty.
        Scaled down, since it is not final, and can be corrected.
        """
        pass

    def _get_expected_nomination_fraction(
        self, extractor: Extractor, actual: bool
    ) -> float | np.ndarray:
        """
        Returns the sum of all nominations.
        If actual:
            - Use the current timestep nominations
        If not actual:
            - Sum the full arrays of nominations
        """
        total_nomination = sum(
            self._get_target(nom, extractor.i, actual)
            for nom in [
                extractor.total_solar_nomination,
                extractor.total_wind_nomination,
                extractor.total_demand_nomination,
            ]
        )
        # single calculation for actual
        if actual:
            return total_nomination * (extractor.i % 60) / 60
        # full array calculation
        indices = np.arange(len(total_nomination))
        return total_nomination * (indices % 60) / 60
