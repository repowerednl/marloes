"""Electrolyser agent with functionality from Repowered's Simon"""
from simon.assets.electrolyser import Electrolyser
from .base import Agent

class ElectrolyserAgent(Agent):
    def __init__(self, config: dict):
        model = Electrolyser(config)

        super().__init__(model)

    def act(self):
        pass

    def observe(self):
        pass

    def learn(self):
        pass