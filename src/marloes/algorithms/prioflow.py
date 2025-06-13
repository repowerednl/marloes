from marloes.algorithms.util import get_net_forecasted_power
from .base import BaseAlgorithm
from collections import defaultdict
import math


class PrioFlow(BaseAlgorithm):
    """
    Priority-based algorithm that solves energy flows using pre-defined PrioFlow.
    """

    __name__ = "PrioFlow"

    def __init__(self, config: dict, evaluate: bool = False):
        """
        Initializes the PrioFlow algorithm.
        """
        super().__init__(config, evaluate)
        self.normalize = False  # No normalization needed for PrioFlow

    @staticmethod
    def _determine_battery_actions(
        net_power: float, batteries: dict
    ) -> dict[str, float]:
        """
        Determines the action for all batteries based on the net power.
        Batteries will now always participate in charging or discharging, based on net power.
        """
        total_capacity = (
            sum(bat["energy_capacity"] for bat in batteries.values()) or 1.0
        )

        battery_actions = {}
        for key, battery in batteries.items():
            # Get share of the battery; weighted net power
            share = battery["energy_capacity"] / total_capacity
            desired_action = -net_power * share
            battery_actions[key] = desired_action / battery["max_power_out"]

        return battery_actions

    def _get_batteries(self, observations: dict) -> dict:
        """
        Extracts the batteries from the observations for the PrioFlow algorithm.
        """
        batteries = defaultdict()
        for key in [handler for handler in observations.keys() if "Battery" in handler]:
            # we need state of charge, capacity and max power out for these batteries
            batteries[key] = {}
            batteries[key]["energy_capacity"] = next(
                handler.asset.energy_capacity
                for handler in self.environment.handlers
                if handler.id == key
            )
            batteries[key]["max_power_out"] = next(
                handler.asset.max_power_out
                for handler in self.environment.handlers
                if handler.id == key
            )
        return batteries

    def get_actions(self, observations: dict, deterministic: bool = False) -> dict:
        """
        No explicit actions are needed as the PrioFlow are predefined.
        The Simon Battery needs setpoints in order to work. To define a baseline we define logical steps for the battery.
        If based on forecasts, the net power is positive, the battery should charge.
        If based on forecasts, the net power is negative, the battery should discharge.
        """
        return self._determine_battery_actions(
            get_net_forecasted_power(observations=observations),
            self._get_batteries(observations),
        )

    def perform_training_steps(self, step: int) -> None:
        """
        Overrides the training step. No learning is required for the priority-based algorithm.
        """
        pass
