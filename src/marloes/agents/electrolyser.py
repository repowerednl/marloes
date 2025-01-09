"""Electrolyser agent with functionality from Repowered's Simon"""

from datetime import datetime
import numpy as np
from simon.assets.electrolyser import Electrolyser
from .base import Agent


class ElectrolyserAgent(Agent):
    def __init__(self, config: dict, start_time: datetime):
        super().__init__(Electrolyser, config, start_time)

    @classmethod
    def get_default_config(cls, config: dict) -> dict:
        """Default configuration for an Electrolyser."""
        return {
            "name": "Electrolyser",
            "conversion_efficiency": 82,
            "max_power_in": config["capacity"],
            "min_power_in": config["capacity"] / 100,
            "slew_rate_up": 100,
            "slew_rate_down": 100,
            "startup_time": 10,
            "anode_pressure": 30,
            "storage_pressure": 50,  # for pipes, for tanks it can go up to 350
        }

    @staticmethod
    def merge_configs(default_config: dict, config: dict) -> dict:
        """Merge the default configuration with user-provided values."""
        merged_config = default_config.copy()
        merged_config.update(config)

        # Enforce constraints with regards to the grid, and get rid of capacity
        merged_config["max_power_in"] = min(
            merged_config.get("max_power_in", np.inf), merged_config.pop("capacity")
        )
        merged_config["min_power_in"] = merged_config["max_power_in"] / 100

        return merged_config

    def map_action_to_setpoint(self, action: float) -> float:
        # Electrolyser has a continous action space, range: [-1, 1]
        if action < 0:
            return self.asset.max_power_in * -action
        else:
            return self.asset.max_power_in * action

    def observe(self):
        pass

    def learn(self):
        pass
