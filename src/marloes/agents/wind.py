"""Wind agent with functionality from Repowered's Simon"""

from datetime import datetime

import numpy as np
import pandas as pd
from simon.assets.supply import Supply

from marloes.data.util import read_series

from .base import Agent


class WindAgent(Agent):
    def __init__(self, config: dict, start_time: datetime, data_config: dict = {}):
        series, forecast = self._get_production_series(config, data_config)
        super().__init__(Supply, config, start_time, series, forecast)

    def _get_production_series(
        self, config: dict, data_config: dict
    ) -> tuple[pd.Series, pd.Series]:
        # Read in the right 1 MWp profile from the wind data
        series = read_series(
            f"Wind_{config.get('location')}.parquet", data_config=data_config
        )

        series *= config["power"]

        # Cap at the AC capacity
        series[series > config["AC"]] = config["AC"]

        # Get forecast
        forecast = read_series(
            f"Wind_{config.pop('location')}.parquet",
            forecast=True,
            data_config=data_config,
        )
        forecast *= config["power"]
        forecast[forecast > config["AC"]] = config["AC"]

        return series, forecast  # kW, kW

    def get_default_config(cls, config: dict, id: str) -> dict:
        """Each subclass must define its default configuration."""
        return {
            "name": id,
            "max_power_out": min(config["AC"], config["power"]),
            "curtailable_by_solver": True,
            "upward_dispatchable": False,
        }

    @staticmethod
    def merge_configs(default_config, config):
        """Merge the default configuration with user-provided values."""
        merged_config = default_config.copy()
        merged_config.update(config)

        merged_config["max_power_out"] = min(
            merged_config.get("max_power_out", np.inf),
            merged_config.pop("power"),
            merged_config.pop("AC"),
        )

        return merged_config

    def map_action_to_setpoint(self, action: float) -> float:
        # Wind has a continous action space, range: [0, 1]
        if action < 0:
            return 0
        else:
            return self.asset.max_power_out * action

    def observe(self):
        pass

    def learn(self):
        pass
