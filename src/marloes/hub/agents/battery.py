""" Battery agent with functionality from Repowered's Simon """
from simon.assets.battery import Battery
from .base import Agent

class BatteryAgent(Agent):
    def __init__(self, config: dict):
        model = Battery(config)

        super().__init__(model)

    def act(self):
        pass

    def observe(self):
        pass

    def learn(self):
        pass

