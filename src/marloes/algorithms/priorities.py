from .base import BaseAlgorithm
from marloes.agents.battery import BatteryAgent


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

    def _get_net_power(self) -> float:
        """
        Looks at the forecasts of each supply and demand agent to calculate the net power.
        TODO: Implement this method. MAR-
        """
        YET_TO_IMPLEMENT = 42
        return YET_TO_IMPLEMENT

    def get_actions(self, observations) -> dict:
        """
        No explicit actions are needed as the priorities are predefined.
        The Simon Battery needs setpoints in order to work. To define a baseline we define logical steps for the battery.
        If based on forecasts, the net power is positive, the battery should charge.
        If based on forecasts, the net power is negative, the battery should discharge.
        """
        batteries = dict()
        # get a list of batteries from the observations (each key is a agent.id), thus it should contain "Battery" in the key
        for key in [agent for agent in observations if "Battery" in agent]:
            # for these batteries we need state of charge and capacity
            batteries[key]["soc"] = observations[key]["state_of_charge"]
            batteries[key]["energy_capacity"] = self.environment.agents[
                key
            ].asset.capacity
        net_power = self._get_net_power()
        print(batteries)
        print(net_power)
        return {bat: 1 for bat in batteries.items()}

    def _train_step(self, observations, rewards, dones, infos) -> None:
        """
        Overrides the training step. No learning is required for the priority-based algorithm.
        """
        pass
