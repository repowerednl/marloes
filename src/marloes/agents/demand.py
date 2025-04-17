"""Demand 'agents' with functionality from Repowered's Simon"""

from datetime import datetime

import numpy as np
import pandas as pd
from simon.assets.demand import Demand

from marloes.data.util import read_series

from .base import Agent


class DemandAgent(Agent):
    def __init__(self, config: dict, start_time: datetime):
        series, forecast = self._get_demand_series(config)
        super().__init__(Demand, config, start_time, series, forecast)

    def _get_demand_series(self, config: dict) -> tuple[pd.Series, pd.Series]:
        # Read in the right demand profile
        series = read_series(f"Demand_{config.get('profile')}.parquet")

        # Scale to the right size
        series = series * config.get("scale", 1)

        # Get forecast
        forecast = read_series(f"Demand_{config.pop('profile')}.parquet", forecast=True)
        forecast = forecast * config.pop("scale", 1)

        # To sum easily for summing forecasts return negative forecast
        forecast = -forecast

        return series, forecast  # kW, kW

    @classmethod
    def get_default_config(cls, config: dict, id: str) -> dict:
        """Each subclass must define its default configuration."""
        return {
            "name": id,
            "max_power_in": np.inf,
            # Should not be curtailed
            "curtailable_by_solver": False,
        }

    def map_action_to_setpoint(self, action: float) -> float:
        # Demand has no setpoints
        pass

    def act(self, action: float, timestamp: datetime) -> None:
        # Demand has no setpoints, so no acting is needed
        pass

    def observe(self):
        pass

    def learn(self):
        pass
