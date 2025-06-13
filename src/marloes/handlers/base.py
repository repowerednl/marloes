import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum

import numpy as np
import pandas as pd
from simon.assets.asset import Asset
from simon.data.asset_data import AssetSetpoint

from marloes.data.util import convert_to_hourly_nomination


class SupplyHandlers(Enum):
    SOLAR = "SolarHandler"
    WIND = "WindHandler"


class DemandHandlers(Enum):
    DEMAND = "DemandHandler"


class StorageHandlers(Enum):
    BATTERY = "BatteryHandler"
    ELECTROLYSER = "ElectrolyserHandler"


class Handler(ABC):
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
        Base Handler class.

        :param asset_class: The SIMON `Asset` class you are wrapping (e.g. Demand, Battery, etc.)
        :param config: Configuration dict for the asset.
        :param start_time: Starting datetime of the simulation.
        :param series: (Optional) the main time series that drives the asset, e.g. actual consumption.
        :param forecast: (Optional) a time series representing a forecast for this asset.
        """
        # Set the handler's ID by keeping count of each asset
        cls_name = self.__class__.__name__
        if cls_name not in Handler._id_counters:
            Handler._id_counters[cls_name] = 0

        self.id = f"{cls_name} {Handler._id_counters[cls_name]}"
        Handler._id_counters[cls_name] += 1

        default_config = self.get_default_config(config, self.id.replace("Handler", ""))
        config = self.merge_configs(default_config, config)

        # Build Simon asset, with optional series
        if series is not None:
            self.asset: Asset = asset(series=series, **config)
        else:
            self.asset: Asset = asset(**config)
        self.asset.load_default_state(start_time)

        # Store the forecast if provided in an efficient format
        if forecast is not None:
            self.forecast = forecast
            self.horizon = 240  # 4 hours for now

            # Also add nomination based on forecast
            self.nominated_volume = convert_to_hourly_nomination(forecast)
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

    def get_state(self, time_stamp: datetime) -> dict:
        """
        Get the current state of the handler. Can be overwritten to also include other information.
        """
        state = self.asset.state.model_dump()

        if self.forecast is not None:
            # Get window of forecast
            end_idx = time_stamp + timedelta(minutes=self.horizon - 1)
            state["forecast"] = self.forecast.loc[time_stamp:end_idx].values.astype(
                np.float32
            )

            # Also include nomination
            hour = time_stamp.replace(minute=0, second=0, microsecond=0)
            state["nomination"] = self.nominated_volume.get(hour, 0.0)

            # only add the attribute to the handler if a forecast is present
            if not hasattr(self, "nomination_fraction"):
                self.nomination_fraction = 0.0

            # reset the fraction at the first minute of the hour
            if time_stamp.minute == 0:
                self.nomination_fraction = 0.0

            # Asset has power - add this to the fraction of the hour
            if state["nomination"] != 0:
                self.nomination_fraction += (state["power"] / state["nomination"]) / 60
            else:
                # logging.warning(
                #     "Nomination is zero; skipping fraction update to avoid division by zero."
                # )
                pass

            # Add the current nomination fraction to the state
            state["nomination_fraction"] = self.nomination_fraction

        # remove 'time' from the state since this is the same for all handlers
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
