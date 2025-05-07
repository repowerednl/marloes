import logging
import math
import os
import time
from collections import defaultdict
from typing import Type

import numpy as np
import pandas as pd
from simon.solver import Model
from simon.assets.demand import Demand
from simon.assets.battery import Battery

from marloes.agents import (
    BatteryAgent,
    GridAgent,
    SolarAgent,
    WindAgent,
    ElectrolyserAgent,
    DemandAgent,
)
from marloes.agents.base import SupplyAgents, StorageAgents, DemandAgents
from marloes.algorithms.util import get_net_forecasted_power
from marloes.data.extensive_data import ExtensiveDataStore

MINUTES_IN_A_YEAR = 525600


class Extractor:
    """
    Extractor class to gather and store simulation metrics.
    """

    def __init__(self, from_model: bool = True, chunk_size: int = 1):
        """
        Initialize the Extractor with preallocated numpy arrays for metrics.
        """
        self.chunk_size = chunk_size
        self.i = 0
        self.start_time = time.time()
        self.size = math.ceil(MINUTES_IN_A_YEAR / self.chunk_size)

        if from_model:
            self._init_empty_arrays()

    def _init_empty_arrays(self):
        # Marl(oes) info
        self.elapsed_time = np.zeros(self.size)
        self.reward = np.zeros(self.size)

        # Reward info
        self.grid_state = np.zeros(self.size)
        self.total_solar_production = np.zeros(self.size)
        self.total_battery_production = np.zeros(self.size)
        self.total_electrolyser_production = np.zeros(self.size)
        self.total_wind_production = np.zeros(self.size)
        self.total_demand = np.zeros(self.size)
        self.total_grid_production = np.zeros(self.size)
        self.total_battery_intake = np.zeros(self.size)

        # Observation info
        self.total_solar_nomination = np.zeros(self.size)
        self.total_wind_nomination = np.zeros(self.size)
        self.total_demand_nomination = np.zeros(self.size)
        self.total_nomination_fraction = np.zeros(self.size)

    def clear(self):
        """Reset the timestep index to zero."""
        self.i = 0
        self._init_empty_arrays()

    def update(self):
        """Increment the timestep index by one."""
        self.i += 1

    def store_reward(self, reward: float) -> None:
        """Save the reward for the current timestep."""
        self.reward[self.i] = reward

    def from_model(self, model: Model) -> None:
        """
        Extract and store metrics from the given simulation model.
        """
        if self.i >= self.size:
            raise IndexError("Extractor has reached its maximum capacity.")

        # Tracking
        self.elapsed_time[self.i] = time.time() - self.start_time

        # Metrics/Reward info
        output_power_data = self.get_current_power_by_type(model)
        self.grid_state[self.i] = list(model.graph.nodes)[0].state.power

        # Emission info
        self.total_solar_production[self.i] = output_power_data.get("Solar", 0.0)
        self.total_battery_production[self.i] = output_power_data.get("Battery", 0.0)
        self.total_electrolyser_production[self.i] = output_power_data.get(
            "Electrolyser", 0.0
        )
        self.total_wind_production[self.i] = output_power_data.get("Wind", 0.0)
        self.total_grid_production[self.i] = output_power_data.get("Grid", 0.0)

        # Demand info (for nomination)
        self.total_demand[self.i] = self._get_total_flow_to_type(model, Demand)
        self.total_battery_intake[self.i] = self._get_total_flow_to_type(model, Battery)

    def from_files(self, uid: int | None = None, dir: str = "results") -> int:
        """
        Extract information from files based on a unique identifier.
        If no identifier is given, the latest identifier is used.
        """
        # If uid is None, extract latest uid from results/uid.txt
        if not uid:
            with open(f"{dir}/uid.txt", "r") as f:
                uid = int(f.read().strip()) - 2

        # Check if there is a config saved with the given uid
        if not os.path.exists(f"{dir}/configs/{uid}.yaml"):
            raise FileNotFoundError(f"No config found with uid {uid}")

        # Loop over each folder in the ./results directory
        # If there is a file in a folder with the given uid (ends in "_uid.npy"), extract the data and save it to
        # an attribute corresponding to the folder name
        # If there is no file with the given uid just skip the folder
        logging.info(f"Extracting data for uid {uid} from {dir}...")
        for folder in os.listdir(dir):
            folder_path = os.path.join(dir, folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith(f"{uid}.npy"):
                        self.__setattr__(folder, np.load(f"{dir}/{folder}/{file}"))

        # Load the results from the extensive extractor if present
        parquet_path = f"{dir}/dataframes/{uid}.parquet"
        if os.path.exists(parquet_path):
            self.extensive_data = pd.read_parquet(parquet_path)
            # identify all grid to demand columns in the data
            grid_to_demand_columns = [
                col
                for col in self.extensive_data.columns
                if "grid_to_demand" in col.lower()
            ]
            # sum the columns (per timestep)
            self.grid_to_demand = self.extensive_data[grid_to_demand_columns].sum(
                axis=1
            )

        return uid

    def from_observations(self, observations: dict) -> None:
        """
        Save the necessary information from the observations.
        #TODO: extend this to save more information from observations
        - nominations
        """
        if self.i >= self.size:
            raise IndexError("Extractor has reached its maximum capacity.")

        nominations = self._get_total_nomination_by_type(observations)
        self.total_solar_nomination[self.i] = nominations["SolarAgent"]
        self.total_wind_nomination[self.i] = nominations["WindAgent"]
        self.total_demand_nomination[self.i] = nominations["DemandAgent"]

        # sum all nomination_fractions in observations together
        for key, value in observations.items():
            if "nomination_fraction" in value:
                self.total_nomination_fraction[self.i] += value["nomination_fraction"]

    def store_loss(self, loss_dict: dict | None) -> None:
        """
        Store each loss in a dedicated array. If loss_dict is None.
        """
        if self.i >= self.size:
            raise IndexError("Extractor has reached its maximum capacity.")

        for loss_key, loss_val in loss_dict.items():
            # Check for attribute existence
            if not hasattr(self, loss_key):
                setattr(self, loss_key, np.zeros(self.size))
            getattr(self, loss_key)[self.i] = loss_val

    @staticmethod
    def _get_total_nomination_by_type(observations: dict) -> dict[str, float]:
        """
        At a timestep, sums the nomination of all Supply and Demand agents
        - SolarAgent
        - WindAgent
        - DemandAgent
        """
        supply_nominations = {agent.value: 0.0 for agent in SupplyAgents}
        demand_nominations = {agent.value: 0.0 for agent in DemandAgents}

        for agent_id, observation in observations.items():
            agent_type = agent_id.split(" ")[0]
            if agent_type in supply_nominations:
                supply_nominations[agent_type] += observation[
                    "nomination"
                ]  # TODO: what if no nomination?
            if agent_type in demand_nominations:
                demand_nominations[agent_type] += observation["nomination"]

        return supply_nominations | demand_nominations

    @staticmethod
    def _get_total_flow_between_types(model: Model, type1: Type, type2: Type) -> float:
        """
        Sum all flows from assets of type1 to assets of type2 in the model.
        """
        total_flow = 0.0
        for (asset1, asset2), flow in model.edge_flow_tracker.items():
            if isinstance(asset1, type1) and isinstance(asset2, type2):
                total_flow += flow
        return total_flow

    @staticmethod
    def _get_total_flow_to_type(model: Model, type1: Type) -> float:
        """
        Sum all flows to assets of type1 in the model.
        """
        total_flow = 0.0
        for (asset1, asset2), flow in model.edge_flow_tracker.items():
            if isinstance(asset2, type1) and asset2.name != "Curtailment":
                total_flow += flow
        return total_flow

    @staticmethod
    def get_current_power_by_type(
        model: Model,
    ) -> dict[str, float]:
        """
        Aggregate the current power production/intake for all assets, grouped by their type.
        """
        output_power_data = defaultdict(float)

        for asset in model.graph.nodes:
            power = asset.state.power
            asset_type = asset.name.split()[0]  # Take generic part of the name
            output_power_data[asset_type] += max(0, power)

        return dict(output_power_data)

    def add_additional_info_from_model(self, model: Model) -> None:
        """
        Extract additional information from the model. Irrelevant for the base Extractor.
        """
        pass

    def get_all_metrics(self):
        """
        Return list of names of all attributes that are a numpy array.
        """
        return [
            attr
            for attr in dir(self)
            if isinstance(getattr(self, attr), np.ndarray)
            and not attr.startswith("_")
            and not attr.endswith("_data")
        ]


class ExtensiveExtractor(Extractor):
    def __init__(self, from_model: bool = True, chunk_size: int = 1):
        super().__init__(from_model, chunk_size)

        # Additional MARL info
        self.action_probability_distribution = np.zeros(self.size)

        # Complete flows/state dataframe
        self.extensive_data = ExtensiveDataStore()

        # Forecasts
        self.solar_forecast = np.zeros(self.size)
        self.wind_forecast = np.zeros(self.size)
        self.demand_forecast = np.zeros(self.size)
        self.net_forecasted_power = np.zeros(self.size)

    def clear(self):
        # Stash the dataframe as part of the clear operation
        super().clear()
        self.extensive_data.stash_chunk()

    def from_observations(self, observations):
        super().from_observations(observations)

        # Fill in the forecast data
        # Since forecast does not change, we can just iteratively add the first value of the forecast series
        # to the forecast array
        for agent_id, observation in observations.items():
            if "forecast" in observation:
                forecast = observation["forecast"]
                if isinstance(forecast, np.ndarray):
                    forecast = forecast[0]
                if agent_id.startswith("SolarAgent"):
                    self.solar_forecast[self.i] = forecast
                elif agent_id.startswith("WindAgent"):
                    self.wind_forecast[self.i] = forecast
                elif agent_id.startswith("DemandAgent"):
                    self.demand_forecast[self.i] = forecast

        self.net_forecasted_power[self.i] = get_net_forecasted_power(observations)

    def add_additional_info_from_model(self, model: Model) -> None:
        """
        Add the flow information after the step to the extensive data.
        """
        self.extensive_data.add_step_data(model)
