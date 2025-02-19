""" Battery agent with functionality from Repowered's Simon """

from datetime import datetime
from functools import partial

import numpy as np
from simon.assets.battery import Battery
from simon.data.battery_data import BatteryState

from .base import Agent


class BatteryAgent(Agent):
    def __init__(self, config: dict, start_time: datetime):
        super().__init__(Battery, config, start_time)

    @classmethod
    def get_default_config(cls, config, id: str) -> dict:
        """Default configuration for a Battery."""
        degradation_function = partial(
            battery_degradation_function,
            capacity=config["energy_capacity"],
            # Default to 7000 cycles
            total_cycles=config.get("total_cycles", 8000),
        )
        return {
            "name": id,
            "max_power_in": config["power"],
            "max_power_out": config["power"],
            "max_state_of_charge": 0.95,  # Assumption: 5% from max and min
            "min_state_of_charge": 0.05,
            "energy_capacity": config["energy_capacity"],
            "ramp_up_rate": config["power"],  # instant
            "ramp_down_rate": config["power"],  # instant
            "efficiency": 0.85,
            "degradation_function": degradation_function,
        }

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
        # Battery has a continous action space, range: [-1, 1]
        if action < 0:
            return self.asset.max_power_in * action
        else:
            return self.asset.max_power_out * action

    def get_state(self, start_idx):
        """
        Also removes the 'is_fcr' key from the state, since this is not relevant for this stage of MARLoes.
        """
        state = super().get_state(start_idx)
        if "is_fcr" in state:
            state.pop("is_fcr")
        return state

    def observe(self):
        pass

    def learn(self):
        pass


def battery_degradation_function(
    time_step: float, state: BatteryState, capacity: float, total_cycles: int
) -> float:
    """
    Simple linear degradation function for a battery.
    Assumptions: total cycles -> 60% battery capacity left
    One cycle = full charge + full discharge = 2 * capacity
    Degradation = degradation per cycle * cycles done
    """
    degradation_per_cycle = (1 - 0.6) / total_cycles
    power_output = time_step * abs(state.power) / 3600
    full_cycle = 2 * capacity
    cycles_done = power_output / full_cycle
    new_degradation = degradation_per_cycle * cycles_done
    return state.degradation + new_degradation
