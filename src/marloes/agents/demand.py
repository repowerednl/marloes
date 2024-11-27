"""Demand 'agents' with functionality from Repowered's Simon"""
from datetime import datetime
from simon.assets.demand import Demand
from .base import Agent
import numpy as np
import pandas as pd


class DemandAgent(Agent):
    def __init__(self, config: dict, start_time: datetime):
        series = self._get_demand_series(config)
        super().__init__(Demand, config, start_time, series)

    def _get_demand_series(self, config: dict):
        # TODO: to be implemented
        return pd.Series()

    def get_default_config(cls, config: dict) -> dict:
        """Each subclass must define its default configuration."""
        return {
            "name": "Demand",
            "max_power_in": np.inf,
            # Should not be curtailed
            "curtailable_by_solver": False,
        }

    def act(self):
        pass

    def observe(self):
        pass

    def learn(self):
        pass
