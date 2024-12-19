"""Electrolyser agent with functionality from Repowered's Simon"""
from datetime import datetime
from simon.assets.electrolyser import Electrolyser
from .base import Agent


class ElectrolyserAgent(Agent):
    def __init__(self, config: dict, start_time: datetime):
        super().__init__(Electrolyser, config, start_time)

    def get_default_config(cls, config: dict) -> dict:
        """Each subclass must define its default configuration."""
        pass

    def map_action_to_setpoint(self, action: float) -> float:
        # Electrolyser has a continous action space, range: [-1, 1]
        if action < 0:
            return self.asset.max_power_in * -action
        else:
            return self.asset.max_power_out * action

    def observe(self):
        pass

    def learn(self):
        pass
