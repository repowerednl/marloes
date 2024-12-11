import numpy as np
from marloes.results.extractor import Extractor

SOLAR_CO2_COEFFICIENT = 0.0
WIND_CO2_COEFFICIENT = 0.0
BATTERY_CO2_COEFFICIENT = 0.0
ELECTROLYSER_CO2_COEFFICIENT = 0.0
GRID_CO2_COEFFICIENT = 0.0


class Reward:
    """
    A generic Reward class that combines multiple sub-penalties/rewards into a total reward.
    """

    def __init__(
        self,
        co2: bool = False,
        ss: bool = False,
        nc: bool = False,
        nb: bool = False,
        lambda_: float = 1.0,
        gamma: float = 1.0,
    ):
        """
        Initializes the Reward instance with specified active sub-penalties/rewards.
        """
        self.co2 = co2
        self.ss = ss
        self.nc = nc
        self.nb = nb
        self.lambda_ = lambda_
        self.gamma = gamma

    def get(
        self,
        extractor: Extractor,
        latest: bool = True,
    ) -> float | np.ndarray:
        """
        Calculate the total reward based on the active sub-rewards.
        `latest` boolean can be set to false to calculate the to
        """
        # Mapping of active sub-rewards
        active_sub_rewards = {
            "CO2": self.co2,
            "SS": self.ss,
            "NC": self.nc,
            "NB": self.nb,
        }

        total_reward = 0.0

        for key, is_active in active_sub_rewards.items():
            if is_active:
                # Get the method for the sub-reward
                method = getattr(self, key.lower(), None)
                if callable(method):
                    reward_value = method(extractor, latest)
                    total_reward += reward_value
                else:
                    raise AttributeError(
                        f"Method '{key.lower()}' not implemented for sub-reward '{key}'."
                    )

        return total_reward

    @staticmethod
    def _get_target(array: np.ndarray, actual: bool = True) -> float | np.ndarray:
        """
        Get the target value/list for the reward calculation.
        """
        if actual:
            return array[:-1]
        else:
            return array

    def co2(self, extractor: Extractor, actual: bool = True) -> float | np.ndarray:
        """
        Calculate the CO2 penalty based on the total CO2 emissions and coefficients.
        """
        solar_penalty = SOLAR_CO2_COEFFICIENT * self._get_target(
            extractor.total_solar_production, actual
        )
        wind_penalty = WIND_CO2_COEFFICIENT * self._get_target(
            extractor.total_wind_production, actual
        )
        battery_penalty = BATTERY_CO2_COEFFICIENT * self._get_target(
            extractor.total_battery_production, actual
        )
        # electrolyser_penalty = ELECTROLYSER_CO2_COEFFICIENT * self._get_target(extractor.total_electrolyser_production, actual)
        grid_penalty = GRID_CO2_COEFFICIENT * self._get_target(
            extractor.total_grid_production, actual
        )

        penalty = solar_penalty + wind_penalty + battery_penalty + grid_penalty
        return penalty

    def ss(self, extractor: Extractor, actual: bool = True) -> float | np.ndarray:
        """
        Calculate the Self-Sufficieny penalty.
        """
        pass

    def nc(self, extractor: Extractor, actual: bool = True) -> float | np.ndarray:
        """
        Calculate the NC penalty.
        """
        pass

    def nb(self, extractor: Extractor, actual: bool = True) -> float | np.ndarray:
        """
        Calculate the NB penalty.
        """
        pass

    def __repr__(self) -> str:
        """
        Returns the string representation of the Reward instance.

        Returns:
            str: String representation.
        """
        active = ", ".join(
            [
                f"{key}={value}"
                for key, value in {
                    "CO2": self.co2,
                    "SS": self.ss,
                    "NC": self.nc,
                    "NB": self.nb,
                }.items()
                if value
            ]
        )
        return (
            f"Reward(active_sub_rewards={{ {active} }}, "
            f"combination_method='{self.combination_method}')"
        )
