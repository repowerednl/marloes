"""
Functions to set up the assets with necessary constraints
"""
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from simon.assets.asset import Asset
from simon.data.asset_data import AssetSetpoint
import pandas as pd


class Agent(ABC):
    _id_counters = {}

    def __init__(
        self, asset: Asset, config: dict, start_time: datetime, series: pd.Series = None
    ):
        # Set the agent's ID by keeping count of each asset
        cls_name = self.__class__.__name__
        if cls_name not in Agent._id_counters:
            Agent._id_counters[cls_name] = 0

        self.id = f"{cls_name} {Agent._id_counters[cls_name]}"
        Agent._id_counters[cls_name] += 1

        default_config = self.get_default_config(config, self.id.replace("Agent", ""))
        config = self.merge_configs(default_config, config)
        if series is not None:
            self.asset: Asset = asset(series=series, **config)
        else:
            self.asset: Asset = asset(**config)
        self.asset.load_default_state(start_time)

    @classmethod
    @abstractmethod
    def get_default_config(cls, config: dict, id: str) -> dict:
        """Each subclass must define its default configuration."""
        pass

    @staticmethod
    def merge_configs(default_config: dict, config: dict) -> dict:
        """Merge the default configuration with user-provided values."""
        merged_config = default_config.copy()  # Start with defaults
        merged_config.update(config)  # Override with provided values
        return merged_config

    @abstractmethod
    def map_action_to_setpoint(self, action: float) -> float:
        pass

    def act(self, action: float, timestamp: datetime) -> None:
        # Map the action to a setpoint value
        value = self.map_action_to_setpoint(action)

        # Create setpoint for the coming minute
        setpoint = AssetSetpoint(
            value=value, start=timestamp, stop=timestamp + timedelta(minutes=1)
        )

        # Set the setpoint
        self.asset.set_setpoint(setpoint)

    @abstractmethod
    def observe(self):
        pass

    @abstractmethod
    def learn(self):
        pass
