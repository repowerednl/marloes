"""Electrolyser agent with functionality from Repowered's Simon"""

from datetime import datetime
import numpy as np
from simon.assets.battery import Battery

from .base import Agent


class ElectrolyserAgent(Agent):
    def __init__(self, config: dict, start_time: datetime):
        super().__init__(Battery, config, start_time)

    @classmethod
    def get_default_config(cls, config: dict) -> dict:
        """Default configuration for an Electrolyser."""
        return {
            "name": "Electrolyser",
            "max_power_in": config["power"],
            "max_power_out": config["power"],
            "capacity": 1,  # x kgH2 / conversion_factor
            "ramp_up_rate": 0.7,  # a certain percentage of capacity
            "ramp_down_rate": 0.7,  # a certain percentage of capacity
            "efficiency_input": 0.6,  # values from Stoff2? or from literature
            "efficiency_output": 0.6,  # values from Stoff2? or from literature
        }

    # SAME AS BATTERY
    @staticmethod
    def merge_configs(default_config: dict, config: dict) -> dict:
        """Merge the default configuration with user-provided values."""
        merged_config = default_config.copy()  # Start with defaults
        merged_config.update(config)  # Override with provided values

        # Enforce constraints with regards to the grid
        merged_config["max_power_in"] = min(
            merged_config.get("max_power_in", np.inf), merged_config["power"]
        )
        merged_config["max_power_out"] = min(
            merged_config.get("max_power_out", np.inf), merged_config.pop("power")
        )

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
