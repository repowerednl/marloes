""" Battery agent with functionality from Repowered's Simon """
from datetime import datetime
from simon.assets.battery import Battery
from simon.data.battery_data import BatteryState
from .base import Agent

class BatteryAgent(Agent):
    def __init__(self, start_time: datetime, config: dict):
        model = Battery(**config)
        model.load_default_state(start_time)

        super().__init__(model)

    def act(self):
        pass

    def observe(self):
        pass

    def learn(self):
        pass

