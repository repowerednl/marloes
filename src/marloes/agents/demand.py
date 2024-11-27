"""Demand 'agents' with functionality from Repowered's Simon"""
from datetime import datetime
from simon.assets.demand import Demand
from .base import Agent


class DemandAgent(Agent):
    def __init__(self, config: dict, start_time: datetime):
        series = self.get_demand_series(config)
        super().__init__(Demand, config, start_time, series)

    def get_demand_series(self, config: dict):
        # TODO: to be implemented
        pass

    def act(self):
        pass

    def observe(self):
        pass

    def learn(self):
        pass
