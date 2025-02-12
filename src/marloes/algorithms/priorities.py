from .base import BaseAlgorithm
from collections import defaultdict
import math


class Priorities(BaseAlgorithm):
    """
    Priority-based algorithm that solves energy flows using pre-defined priorities.
    """

    __name__ = "Priorities"

    def __init__(self, config: dict):
        """
        Initializes the Priorities algorithm.
        """
        super().__init__(config)

    @staticmethod
    def _determine_battery_actions(
        net_power: float, batteries: dict, soc_threshold: float = 0.3
    ) -> dict[str, float]:
        """
        Determines the action for the batteries based on the net power and the battery state (SOC and capacity).
        Batteries should first qualify (above certain SOC).
        Then depending on the capacities, create actions to ratio > TODO
        """
        qualified_batteries = {
            key: battery
            for key, battery in batteries.items()
            if (battery["soc"] > soc_threshold and net_power < 0) or net_power > 0
        }
        ratio = 1 / len(qualified_batteries) if len(qualified_batteries) > 0 else 0
        active_batteries = {
            # charge or discharge the battery based on the net power
            key: (math.copysign(1.0, net_power) * ratio)
            * qualified_batteries[key]["energy_capacity"]
            for key in qualified_batteries.keys()
        }
        # return the active batteries, and the other batteries to 0
        return {key: active_batteries.get(key, 0) for key in batteries.keys()}

    def _get_batteries(self, observations: dict) -> dict:
        """
        Extracts the batteries from the observations for the Priorities algorithm.
        """
        batteries = defaultdict()
        for key in [agent for agent in observations.keys() if "Battery" in agent]:
            # we need state of charge and capacity for these batteries
            batteries[key] = {}
            batteries[key]["soc"] = observations[key]["state_of_charge"]
            batteries[key]["energy_capacity"] = next(
                agent.asset.energy_capacity
                for agent in self.environment.agents
                if agent.id == key
            )
        return batteries

    def get_actions(self, observations: dict) -> dict:
        """
        No explicit actions are needed as the priorities are predefined.
        The Simon Battery needs setpoints in order to work. To define a baseline we define logical steps for the battery.
        If based on forecasts, the net power is positive, the battery should charge.
        If based on forecasts, the net power is negative, the battery should discharge.
        """
        return self._determine_battery_actions(
            self._get_net_forecasted_power(observations=observations),
            self._get_batteries(observations),
        )

    def _train_step(self, observations, rewards, dones, infos) -> None:
        """
        Overrides the training step. No learning is required for the priority-based algorithm.
        """
        pass
