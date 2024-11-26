"""Solar agent with functionality from Repowered's Simon"""
from simon.assets.supply import Supply
from .base import Agent

class SolarAgent(Agent):
    def __init__(self, config: dict):
        model = Supply(config)

        super().__init__(model)

    def act(self):
        pass

    def observe(self):
        pass

    def learn(self):
        pass