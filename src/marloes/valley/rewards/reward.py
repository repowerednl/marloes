from datetime import datetime
import logging

import numpy as np

from marloes.results.extractor import Extractor
from marloes.valley.rewards.subrewards import (
    CO2SubReward,
    NBSubReward,
    NCSubReward,
    SSSubReward,
    SubReward,
    NESubReward,
    TESTSubReward,
)
from marloes.valley.rewards.subrewards.balance import BalanceSubReward


class Reward:
    """
    A generic Reward class that combines multiple sub-penalties/rewards into a total reward.
    """

    VALID_SUB_REWARDS = {
        "CO2": CO2SubReward,
        "SS": SSSubReward,
        "NC": NCSubReward,
        "NB": NBSubReward,
        "NE": NESubReward,
        "TEST": TESTSubReward,
        "BA": BalanceSubReward,
    }

    def __init__(self, config: dict, actual: bool = True, **kwargs):
        """
        Initializes the Reward instance with specified sub-penalties/rewards.
        """
        self.actual = actual
        self.sub_rewards: dict[str, SubReward] = {
            key.upper(): self.VALID_SUB_REWARDS[key.upper()](config, **value)
            for key, value in kwargs.items()
            if key.upper() in self.VALID_SUB_REWARDS
        }
        logging.info(
            f"Initialized Reward with sub-rewards: {[sub for sub in self.sub_rewards if self.sub_rewards[sub].active]}"
        )

        # Track the net grid state if ss or nb is active and actual is True
        self.net_grid_state = 0

    def get(self, extractor: Extractor, time_stamp: datetime) -> float | np.ndarray:
        """
        Calculate the total reward based on the active sub-rewards.
        """
        if self.actual:
            total_reward = 0

            # For the SS subreward
            self.net_grid_state += extractor.grid_state[extractor.i]
            input_dict = {
                "net_grid_state": self.net_grid_state,
                "grid_state": extractor.grid_state[extractor.i],
                "total_demand": extractor.total_demand[extractor.i],
                "total_battery_intake": extractor.total_battery_intake[extractor.i],
            }

        else:
            total_reward = np.zeros(len(extractor.grid_state))

        for _, sub_reward in self.sub_rewards.items():
            if sub_reward.active:
                total_reward += sub_reward.scaling_factor * sub_reward.calculate(
                    extractor,
                    actual=self.actual,
                    input_dict=input_dict if self.actual else None,
                    time_stamp=time_stamp,
                )

        return total_reward

    def __repr__(self):
        sub_rewards_state = {
            key: {"active": reward.active, "scaling_factor": reward.scaling_factor}
            for key, reward in self.sub_rewards.items()
        }
        return f"Reward(actual={self.actual}, sub_rewards={sub_rewards_state})"
