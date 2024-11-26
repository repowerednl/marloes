"""Demand 'agents' with functionality from Repowered's Simon"""
from datetime import datetime
from simon.assets.demand import Demand
from .base import Agent


class DemandAgent(Agent):
    def __init__(self, start_time: datetime, config: dict):
        model = Demand(**config)
        model.load_default_state(time=start_time)

        super().__init__(model)

    def act(self):
        pass

    def observe(self):
        pass

    def learn(self):
        pass