"""Wind agent with functionality from Repowered's Simon"""
from datetime import datetime
from simon.assets.supply import Supply
from .base import Agent


class WindAgent(Agent):
    def __init__(self, config: dict, start_time: datetime):
        series = self._get_production_series(config)
        super().__init__(Supply, config, start_time, series)

    def _get_production_series(self, config: dict):
        # TODO: to be implemented
        pass

    def get_default_config(cls, config: dict) -> dict:
        """Each subclass must define its default configuration."""
        pass

    def act(self):
        pass

    def observe(self):
        pass

    def learn(self):
        pass
