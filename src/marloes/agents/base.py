import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from simon.assets.asset import Asset
from simon.data.asset_data import AssetSetpoint

from marloes.data.util import convert_to_hourly_nomination


class Agent(ABC):
    _id_counters = {}

    def __init__(
        self,
        asset: Asset,
        config: dict,
        start_time: datetime,
        series: pd.Series = None,
        forecast: pd.Series = None,
    ):
        """
        Base Agent class.

        :param asset_class: The SIMON `Asset` class you are wrapping (e.g. Demand, Battery, etc.)
        :param config: Configuration dict for the asset.
        :param start_time: Starting datetime of the simulation.
        :param series: (Optional) the main time series that drives the asset, e.g. actual consumption.
        :param forecast: (Optional) a time series representing a forecast for this asset.
        """
        # Set the agent's ID by keeping count of each asset
        cls_name = self.__class__.__name__
        if cls_name not in Agent._id_counters:
            Agent._id_counters[cls_name] = 0

        self.id = f"{cls_name} {Agent._id_counters[cls_name]}"
        Agent._id_counters[cls_name] += 1

        default_config = self.get_default_config(config, self.id.replace("Agent", ""))
        config = self.merge_configs(default_config, config)

        # Build Simon asset, with optional series
        if series is not None:
            self.asset: Asset = asset(series=series, **config)
        else:
            self.asset: Asset = asset(**config)
        self.asset.load_default_state(start_time)

        # Store the forecast if provided in an efficient format
        if forecast is not None:
            self.forecast: np.ndarray = forecast.values.astype(np.float32)
            self.horizon = 1440  # 24 hours for now

            # Also add nomination based on forecast
            self.nominated_volume: np.ndarray = convert_to_hourly_nomination(
                self.forecast
            )
        else:
            self.forecast = None

        logging.info(f"{self.id} initialized...")

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

    def get_state(self, start_idx: int) -> dict:
        """
        Get the current state of the agent. Can be overwritten to also include other information.
        """
        state = self.asset.state.model_dump()

        if self.forecast is not None:
            end_idx = start_idx + self.horizon
            end_idx = min(
                end_idx, len(self.forecast)
            )  # Ensure we don't go out of bounds
            state["forecast"] = self.forecast[
                start_idx:end_idx
            ]  # Numpy slicing is O(1)

            # Also include nomination
            hour_idx = start_idx // 60  # 60 minutes in an hour
            hour_idx = min(hour_idx, len(self.nominated_volume) - 1)  # safety
            # Add the current hour's nomination to the state
            state["nomination_kW"] = float(self.nominated_volume[hour_idx])

        # remove 'time' from the state since this is the same for all agents
        if "time" in state:
            state.pop("time")

        return state

    @abstractmethod
    def map_action_to_setpoint(self, action: float) -> float:
        pass

    def act(self, action: float, timestamp: datetime) -> None:
        """
        Convert the action into a setpoint that is valid for the next time interval
        and set it on the underlying SIMON asset.
        """
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
