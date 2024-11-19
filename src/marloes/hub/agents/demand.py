"""Demand 'agents' with functionality from Repowered's Simon"""
from simon.assets.demand import Demand
from .base import Agent

class DemandAgent(Agent):
    def __init__(self, config: dict):
        model = Demand(config)

        super().__init__(model)

    def act(self):
        pass

    def observe(self):
        pass

    def learn(self):
        pass