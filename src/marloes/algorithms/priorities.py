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
        net_power: float, batteries: dict
    ) -> dict[str, float]:
        """
        Determines the action for all batteries based on the net power.
        Batteries will now always participate in charging or discharging, based on net power.
        """
        num_batteries = len(batteries)
        ratio = 1 / num_batteries if num_batteries > 0 else 0

        battery_actions = {
            key: math.copysign(1.0, net_power) * ratio * battery["energy_capacity"]
            for key, battery in batteries.items()
        }
        return battery_actions

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

    def perform_training_steps(self, step: int) -> None:
        """
        Overrides the training step. No learning is required for the priority-based algorithm.
        """
        pass
