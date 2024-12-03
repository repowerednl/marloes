""" Battery agent with functionality from Repowered's Simon """
from datetime import datetime
from functools import partial
from simon.assets.battery import Battery
from simon.data.battery_data import BatteryState
from .base import Agent


class BatteryAgent(Agent):
    def __init__(self, config: dict, start_time: datetime):
        super().__init__(Battery, config, start_time)

    @classmethod
    def get_default_config(cls, config) -> dict:
        """Default configuration for a Battery."""
        if not config.get("max_power_in") or not config.get("energy_capacity"):
            raise ValueError(
                "Battery configuration minimally requires 'max_power_in' and 'energy_capacity'"
            )
        degradation_function = partial(
            battery_degradation_function,
            capacity=config["energy_capacity"],
            # Default to 7000 cycles
            total_cycles=config.get("total_cycles", 7000),
        )
        return {
            "name": "Battery",
            "max_power_in": config["max_power_in"],
            "max_power_out": config["max_power_in"],
            "max_state_of_charge": 0.95,  # Assumption: 5% from max and min
            "min_state_of_charge": 0.05,
            "energy_capacity": config["energy_capacity"],
            "ramp_up_rate": config["max_power_in"],  # instant
            "ramp_down_rate": config["max_power_in"],  # instant
            "efficiency": 0.85,
            "degradation_function": degradation_function,
        }

    def act(self, action: float):
        pass

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
