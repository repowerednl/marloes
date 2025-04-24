import numpy as np

from marloes.results.extractor import Extractor
from marloes.valley.rewards.subrewards.base import SubReward

import logging


class NESubReward(SubReward):
    """
    NE: Nomination Error
    Sub-reward that incentivizes following nomination. Any deviation from the nomination is penalized.
    """

    name = "NE"

    def __init__(
        self,
        active: bool = False,
        scaling_factor: float = 1.0,
        intermediate_scaling_factor: float = None,
    ):
        """
        Initializes the NESubReward instance with activation and scaling properties.
        Initializes allows for a different scaling factor for the intermediate penalty;
        - defaults to 1/60 of the normal scaling factor.
        """
        super().__init__(active, scaling_factor)
        if intermediate_scaling_factor is None:
            self.intermediate_scaling_factor = scaling_factor / 60
        else:
            self.intermediate_scaling_factor = intermediate_scaling_factor

        if self.intermediate_scaling_factor > scaling_factor:
            logging.warning(
                "Intermediate scaling factor exceeds normal scaling factor."
            )

    def calculate(
        self, extractor: Extractor, actual: bool, **kwargs
    ) -> float | np.ndarray:
        """
        Calculates the difference between the power and nomination of all supply (wind and solar) assets.
        Is calculated every hour with the difference between the mean of the total production and the nomination.
        """
        if actual:
            return (
                self._calculate_penalty(
                    extractor, slice(extractor.i - 60, extractor.i), actual=True
                )
                if self._hour_has_passed(extractor.i)
                else self._calculate_intermediate_penalty(extractor, actual=True)
            )
        # Calculate intermediate penalties and full penalties only at required timesteps
        reward_array = self._calculate_intermediate_penalty(extractor, actual=False)
        full_penalty_indices = np.arange(60, len(reward_array), 60)
        full_penalties = [
            self._calculate_penalty(
                extractor, slice(t - 60, t), actual=True
            )  # needs to be True to use the slices
            for t in full_penalty_indices
        ]
        reward_array[full_penalty_indices] = full_penalties

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
        demand = self._get_target(extractor.total_demand, time_slice, actual)
        demand_nomination = self._get_target(
            extractor.total_demand_nomination, time_slice, actual
        )
        # add all production together
        total_production = solar_production + wind_production + demand
        # add all nominations together
        total_nomination = solar_nomination + wind_nomination + demand_nomination
        # calculate the penalty
        penalty = -(
            abs(sum(total_production) - sum(total_nomination)) * self.scaling_factor
        )
        return penalty

    @staticmethod
    def _hour_has_passed(i: int) -> bool:
        return i % 60 == 0 and i != 0

    def _calculate_intermediate_penalty(
        self, extractor: Extractor, actual: bool
    ) -> float | np.ndarray:
        """
        Returns the difference between actual nomination_fraction and the expected nomination_fraction as a small penalty.
        Scaled down, since it is not final, and can be corrected.
        """
        nomination_fraction = self._get_target(
            extractor.total_nomination_fraction, extractor.i, actual
        )
        # the expected nomination fraction is the sum of all nominations / 60 * (i % 60)
        expected_nomination_fraction = self._get_expected_nomination_fraction(
            extractor, actual
        )
        return (
            -abs(nomination_fraction - expected_nomination_fraction)
            * self.intermediate_scaling_factor
        )

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
