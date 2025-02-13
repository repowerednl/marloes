"""Electrolyser agent with functionality from Repowered's Simon"""

from datetime import datetime
import numpy as np
from simon.assets.battery import Battery
from simon.data.battery_data import BatteryState

from functools import partial
from .base import Agent


class ElectrolyserAgent(Agent):
    def __init__(self, config: dict, start_time: datetime):
        super().__init__(Battery, config, start_time)
        # the internal clock is to keep track of the start-up time of the electrolyser
        self.start_up_time = 5  # value from PEM(EC)
        self.internal_clock = 0
        self.discharge_rate = 1 - (1 / (2 * 24 * 3600))  # 2 days to 0, from Stoff2

    @classmethod
    def get_default_config(cls, config: dict, id: str) -> dict:
        """Default configuration for an Electrolyser."""
        degradation_function = partial(
            electrolyser_degradation_function,
            max_capacity=config["energy_capacity"],
            # default to 80000 hours (60000-100000, from PEM(EC))
            total_lifetime_hours=config.get("lifetime_hours", 80000),
        )
        return {
            "name": id,
            "max_power_in": config["power"],
            "max_power_out": config["power"],
            "max_state_of_charge": 0.95,
            "min_state_of_charge": 0.05,
            "energy_capacity": config["energy_capacity"],  # x kgH2 * conversion_factor
            "ramp_up_rate": 0.4,  # a certain percentage of capacity (Reducing ramp up rate even more to account for start-up time)
            "ramp_down_rate": 0.4,  # a certain percentage of capacity
            "input_efficiency": 0.7,  # values from Stoff2
            "output_efficiency": 0.7,  # values from Stoff2
            "degradation_function": degradation_function,
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
        if action <= 0:
            # reset the internal clock
            self.internal_clock = 0
            return self.asset.max_power_in * action
        else:
            # increment the internal clock
            self.internal_clock += 1
            return (
                self.asset.max_power_out * action
                if self.internal_clock > self.start_up_time
                else 0
            )

    def _loss_discharge(self) -> float:
        """
        Discharge the battery with a given power, the power in the electrolyser has a low 'loss' rate.
        The current power can stay up to 2 days in the electrolyser, reduce the power linearly to 0 in 2 days.
        Function should be called after step() is called.
        """
        # get the current state of the battery, set a new state with the slight loss
        old_state = self.asset.state
        new_state = BatteryState(
            power=old_state.power,
            time=old_state.time,
            state_of_charge=old_state.state_of_charge * self.discharge_rate,
            degradation=old_state.degradation,
            is_fcr=old_state.is_fcr,
        )
        self.asset.set_state(new_state)

    def observe(self):
        pass

    def learn(self):
        pass


def electrolyser_degradation_function(
    time_step: float,
    state: BatteryState,
    max_capacity: float,
    total_lifetime_hours: float,
) -> float:
    """
    Simple linear degradation function for an electrolyzer.
    Assumptions:
    - Total lifetime: 60% efficiency left after `total_lifetime_hours` of operation.
    - Degradation occurs based on operational hours and relative load.
    - Operational load is normalized to maximum capacity.
    Source: IEA - The Future of Hydrogen
    """
    degradation_per_hour = (1 - 0.6) / total_lifetime_hours
    operational_load = (
        abs(state.power) / max_capacity
    )  # Normalize power to max capacity
    hours_operated = time_step / 3600 * operational_load  # Scaled by load factor
    new_degradation = degradation_per_hour * hours_operated
    return state.degradation + new_degradation
