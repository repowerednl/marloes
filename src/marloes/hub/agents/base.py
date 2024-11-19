"""
Functions to set up the assets with necessary constraints
"""
from abc import ABC, abstractmethod
from simon.assets.asset import Asset


class Agent(ABC):
    def __init__(self, asset: Asset):
        self.model = asset

    @abstractmethod
    def act(self):
        pass

    @abstractmethod
    def observe(self):
        pass

    @abstractmethod
    def learn(self):
        pass