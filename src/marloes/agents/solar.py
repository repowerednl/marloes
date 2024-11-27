"""Solar agent with functionality from Repowered's Simon"""
from datetime import datetime
from marloes.data.util import read_series
from simon.assets.supply import Supply
from .base import Agent
import pandas as pd


class SolarAgent(Agent):
    def __init__(self, config: dict, start_time: datetime):
        series = self._get_production_series(config)
        super().__init__(Supply, config, start_time, series)

    def _get_production_series(self, config: dict) -> pd.Series:
        # Read in the right 1 MWp profile from the solar data
        series = read_series(f"Solar_{config['orientation']}")

        # Scale to the right size
        series = series * config["DC"] / 1000  # from kWp to MWp

        # Cap at the AC capacity
        series[series > config["AC"]] = config["AC"]

        return series

    def get_default_config(cls, config: dict) -> dict:
        """Each subclass must define its default configuration."""
        pass

    def act(self):
        pass

    def observe(self):
        pass

    def learn(self):
        pass
