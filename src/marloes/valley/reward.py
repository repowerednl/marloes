from dataclasses import asdict, dataclass
import numpy as np
from marloes.results.extractor import Extractor
import logging


@dataclass
class SubReward:
    """
    Represents an individual sub-reward with activation and scaling properties.
    """

    active: bool = False
    scaling_factor: float = 1.0


class Reward:
    """
    A generic Reward class that combines multiple sub-penalties/rewards into a total reward.
    """

    EMISSION_COEFFICIENTS = {
        "solar": 0,
        "wind": 0,
        "battery": 0,
        "electrolyser": 0,
        "grid": 0,
    }

    VALID_SUB_REWARDS = {"CO2", "SS", "NC", "NB"}

    def __init__(
        self,
        actual: bool = True,
        **kwargs,
    ):
        """
        Initializes the Reward instance with specified sub-penalties/rewards:
            kwargs (dict): Key-value pairs for each sub-reward's activation and scaling factor.
        """
        self.actual = actual
        self.sub_rewards = {
            key.upper(): SubReward(**value)
            for key, value in kwargs.items()
            if key.upper() in self.VALID_SUB_REWARDS
        }
        logging.info(f"Initialized Reward with sub-rewards: {self.sub_rewards}")

        # Track the net grid state if ss or nb is active and actual is True
        self.prev_net_grid_state = 0

    def get(
        self,
        extractor: Extractor,
    ) -> float | np.ndarray:
        """
        Calculate the total reward based on the active sub-rewards.
        """
        if self.actual:
            total_reward = 0
            self.prev_net_grid_state += extractor.grid_state[extractor.i]
        else:
            total_reward = np.zeros(len(extractor.grid_state))

        for key, sub_reward in self.sub_rewards.items():
            if sub_reward.active:
                # Get the method for the sub-reward
                method = getattr(self, f"get_{key.lower()}", None)
                if callable(method):
                    total_reward += sub_reward.scaling_factor * method(extractor)
                else:
                    raise AttributeError(
                        f"Method 'get_{key.lower()}' not implemented for sub-reward '{key}'."
                    )

        return total_reward

    def _get_target(self, array: np.ndarray, i: int) -> float | np.ndarray:
        """
        Get the target value/list for the reward calculation.
        """
        if self.actual:
            return array[i]
        else:
            return array

    def get_co2(self, extractor: Extractor) -> float | np.ndarray:
        """
        Calculate the CO2 penalty based on the total CO2 emissions and coefficients; penalizes the emissions as a result of each
        asset's production at time t.
        """
        i = extractor.i
        penalties = [
            self.EMISSION_COEFFICIENTS["solar"]
            * self._get_target(extractor.total_solar_production, i),
            self.EMISSION_COEFFICIENTS["wind"]
            * self._get_target(extractor.total_wind_production, i),
            self.EMISSION_COEFFICIENTS["battery"]
            * self._get_target(extractor.total_battery_production, i),
            # self.EMISSION_COEFFICIENTS["electrolyser"] * self._get_target(extractor.total_electrolyser_production, i),
            self.EMISSION_COEFFICIENTS["grid"]
            * self._get_target(extractor.total_grid_production, i),
        ]

        return -sum(penalties)

    def get_ss(self, extractor: Extractor) -> float | np.ndarray:
        """
        Calculate the Self-Sufficieny penalty, which penalizes the negative deviation from self-
        sufficiency by considering the negative cumulative net electricity taken from the national
        power network so far.
        """
        if self.actual:
            return -max(0, self.prev_net_grid_state)
        return -np.maximum(0, np.cumsum(extractor.grid_state))

    def get_nc(self, extractor: Extractor) -> float | np.ndarray:
        """
        Calculate the NC penalty, which penalizes the electricity given to the national
        power network at a certain time-step, as that could increase chances of net-congestion.
        """
        grid_state = self._get_target(extractor.grid_state, extractor.i)
        return np.minimum(0, grid_state)

    def get_nb(self, extractor: Extractor) -> float | np.ndarray:
        """
        Calculate the NB penalty. Incentive for net-positive energy balance: the more energy
        produced than consumed, the higher the reward. Also measures the energy balance by
        grid state.
        """
        if self.actual:
            return -self.prev_net_grid_state
        return -np.cumsum(extractor.grid_state)

    def __repr__(self):
        sub_rewards_state = {
            key: asdict(reward) for key, reward in self.sub_rewards.items()
        }
        return f"Reward(actual={self.actual}, sub_rewards={sub_rewards_state})"
