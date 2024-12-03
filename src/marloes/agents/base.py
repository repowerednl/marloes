"""
Functions to set up the assets with necessary constraints
"""
from abc import ABC, abstractmethod
from datetime import datetime
from simon.assets.asset import Asset
import pandas as pd


class Agent(ABC):
    def __init__(
        self, asset: Asset, config: dict, start_time: datetime, series: pd.Series = None
    ):
        default_config = self.get_default_config(config)
        config = self.merge_configs(default_config, config)
        if series is not None:
            self.asset = asset(series=series, **config)
        else:
            self.asset = asset(**config)
        self.asset.load_default_state(start_time)

    @classmethod
    @abstractmethod
    def get_default_config(cls, config: dict) -> dict:
        """Each subclass must define its default configuration."""
        pass

    @staticmethod
    def merge_configs(default_config: dict, config: dict) -> dict:
        """Merge the default configuration with user-provided values."""
        merged_config = default_config.copy()  # Start with defaults
        merged_config.update(config)  # Override with provided values
        return merged_config

    @abstractmethod
    def act(self, action: float):
        pass

    @abstractmethod
    def observe(self):
        pass

    @abstractmethod
    def learn(self):
        pass
