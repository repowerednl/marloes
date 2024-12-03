"""Demand 'agents' with functionality from Repowered's Simon"""
from datetime import datetime
from simon.assets.demand import Demand

from marloes.data.util import read_series
from .base import Agent
import numpy as np
import pandas as pd


class DemandAgent(Agent):
    def __init__(self, config: dict, start_time: datetime):
        series = self._get_demand_series(config)
        super().__init__(Demand, config, start_time, series)

    def _get_demand_series(self, config: dict):
        # Read in the right demand profile
        series = read_series(f"Demand_{config['profile']}.parquet")

        # Scale to the right size
        series = series * config.get("scale", 1)

        # Remove used arguments from config
        config.pop("profile", None)
        config.pop("scale", None)

        return series

    def get_default_config(cls, config: dict) -> dict:
        """Each subclass must define its default configuration."""
        return {
            "name": "Demand",
            "max_power_in": np.inf,
            # Should not be curtailed
            "curtailable_by_solver": False,
        }

    def act(self, action: float):
        pass

    def observe(self):
        pass

    def learn(self):
        pass
