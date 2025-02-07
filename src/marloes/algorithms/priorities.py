from .base import BaseAlgorithm
from collections import defaultdict


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
    def _get_net_power(observations: dict, period: int = 60) -> float:
        """
        Looks at the forecasts of each supply and demand agent to calculate the net power.
        Sum the forecasts of the next period, period is in minutes, defaults to 60 min (1 hour).
        """

        forecasts = [
            observations[agent]["forecast"]
            for agent in observations.keys()
            if "forecast" in observations[agent]
        ]
        # TODO: What does forecast look like? pd.Series? dict? list?
        # select only the relevant period
        return sum(forecast[min(period, len(forecast)) - 1] for forecast in forecasts)

    @staticmethod
    def _determine_battery_actions(
        net_power: float, batteries: dict
    ) -> dict[str, float]:
        """
        Determines the action for the batteries based on the net power and the battery state (SOC and capacity).
        Batteries should first qualify (above certain SOC).
        Then depending on the capacities, create actions to ratio.
        """
        qualified_batteries = {
            key: battery for key, battery in batteries.items() if battery["soc"] > 0.4
        }
        ratio = 1 / len(qualified_batteries) if len(qualified_batteries) > 0 else 0
        if net_power > 0:
            # charge the batteries
            return {
                key: ratio * qualified_batteries[key]["energy_capacity"]
                for key in qualified_batteries.keys()
            }
        elif net_power < 0:
            # discharge the batteries
            return {
                key: -ratio * qualified_batteries[key]["energy_capacity"]
                for key in qualified_batteries.keys()
            }
        return {key: 0 for key in batteries.keys()}

    def get_actions(self, observations: dict) -> dict:
        """
        No explicit actions are needed as the priorities are predefined.
        The Simon Battery needs setpoints in order to work. To define a baseline we define logical steps for the battery.
        If based on forecasts, the net power is positive, the battery should charge.
        If based on forecasts, the net power is negative, the battery should discharge.
        """
        batteries = defaultdict()
        # get a list of batteries from the observations (each key is a agent.id), thus it should contain "Battery" in the key
        for key in [agent for agent in observations.keys() if "Battery" in agent]:
            batteries[key] = {}
            # for these batteries we need state of charge and capacity
            batteries[key]["soc"] = observations[key]["state_of_charge"]
            batteries[key]["energy_capacity"] = next(
                agent.asset.energy_capacity
                for agent in self.environment.agents
                if agent.id == key
            )
        return self._determine_battery_actions(
            self._get_net_power(observations=observations), batteries
        )

    def _train_step(self, observations, rewards, dones, infos) -> None:
        """
        Overrides the training step. No learning is required for the priority-based algorithm.
        """
        pass
