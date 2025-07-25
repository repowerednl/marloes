"""Solar agent with functionality from Repowered's Simon"""

from datetime import datetime

import numpy as np
import pandas as pd
from simon.assets.supply import Supply

from marloes.data.util import read_series

from .base import Agent


class SolarAgent(Agent):
    def __init__(self, config: dict, start_time: datetime, data_config: dict = {}):
        series, forecast = self._get_production_series(config, data_config)
        super().__init__(Supply, config, start_time, series, forecast)

    def _get_production_series(
        self, config: dict, data_config: dict
    ) -> tuple[pd.Series, pd.Series]:
        # Read in the right 1 MWp profile from the solar data
        series = read_series(
            f"Solar_{config.get('orientation')}.parquet", data_config=data_config
        )

        # Scale to the right size
        series = series * config["DC"] / 1000  # from kWp to MWp

        # Cap at the AC capacity
        series[series > config["AC"]] = config["AC"]

        # Get forecast
        forecast = read_series(
            f"Solar_{config.pop('orientation')}.parquet",
            forecast=True,
            data_config=data_config,
        )
        forecast = forecast * config["DC"] / 1000
        forecast[forecast > config["AC"]] = config["AC"]

        return series, forecast  # kW, kW

    def get_default_config(cls, config: dict, id: str) -> dict:
        """Each subclass must define its default configuration."""
        return {
            "name": id,
            "max_power_out": min(config["AC"], config["DC"]),
            # Solar parks are curtailable
            "curtailable_by_solver": True,
            "upward_dispatchable": False,
        }

    @staticmethod
    def merge_configs(default_config: dict, config: dict) -> dict:
        """Merge the default configuration with user-provided values."""
        merged_config = default_config.copy()  # Start with defaults
        merged_config.update(config)  # Override with provided values

        # Enforce constraints with regards to the grid
        merged_config["max_power_out"] = min(
            merged_config.get("max_power_out", np.inf),
            merged_config.pop("AC"),
            merged_config.pop("DC"),
        )

        return merged_config

    def map_action_to_setpoint(self, action: float) -> float:
        # Solar has a continous action space, range: [0, 1]
        if action < 0:
            return 0
        else:
            return self.asset.max_power_out * action

    def observe(self):
        pass

    def learn(self):
        pass
