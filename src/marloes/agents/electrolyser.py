"""Electrolyser agent with functionality from Repowered's Simon"""
from datetime import datetime
from simon.assets.electrolyser import Electrolyser
from .base import Agent


class ElectrolyserAgent(Agent):
    def __init__(self, config: dict, start_time: datetime):
        super().__init__(Electrolyser, config, start_time)

    def act(self):
        pass

    def observe(self):
        pass

    def learn(self):
        pass
