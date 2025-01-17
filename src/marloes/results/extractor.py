import logging
import math
import os
import time
from collections import defaultdict
from typing import Type

import numpy as np
import pandas as pd
from simon.solver import Model

from marloes.agents import BatteryAgent, GridAgent, SolarAgent
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
            # Marl(oes) info
            self.elapsed_time = np.zeros(self.size)
            self.loss = np.zeros(self.size)

            # Reward info
            self.grid_state = np.zeros(self.size)
            self.total_solar_production = np.zeros(self.size)
            self.total_battery_production = np.zeros(self.size)
            self.total_wind_production = np.zeros(self.size)
            self.total_grid_production = np.zeros(self.size)

    def clear(self):
        """Reset the timestep index to zero."""
        self.i = 0

    def update(self):
        """Increment the timestep index by one."""
        self.i += 1

    def from_model(self, model: Model) -> None:
        """
        Extract and store metrics from the given simulation model.
        """
        if self.i >= self.size:
            raise IndexError("Extractor has reached its maximum capacity.")

        # Tracking
        self.elapsed_time[self.i] = time.time() - self.start_time

        # Marl(oes) info
        # TODO: Implement loss tracking
        # self.loss[self.i] = loss

        # Metrics/Reward info
        output_power_data = self.get_current_power_by_type(model)
        self.grid_state[self.i] = list(model.graph.nodes)[0].state.power

        # Emission info
        self.total_solar_production[self.i] = output_power_data.get(
            SolarAgent.__name__, 0.0
        )
        self.total_battery_production[self.i] = output_power_data.get(
            BatteryAgent.__name__, 0.0
        )
        # self.total_wind_production[self.i] = output_power_data.get(WindAgent.__name__, 0.0)
        self.total_grid_production[self.i] = output_power_data.get(
            GridAgent.__name__, 0.0
        )
        self.update()

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

        return uid

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


class ExtensiveExtractor(Extractor):
    def __init__(self, from_model: bool = True, chunk_size: int = 1):
        super().__init__(from_model, chunk_size)

        # Additional MARL info
        self.action_probability_distribution = np.zeros(self.size)

        # Complete flows/state dataframe
        self.extensive_data = ExtensiveDataStore()

    def clear(self):
        # Stash the dataframe as part of the clear operation
        super().clear()
        self.extensive_data.stash_chunk()

    def from_model(self, model: Model) -> None:
        """
        Extract and store metrics from the simulation model,
        including detailed state/flow information.
        """
        super().from_model(model)
        self.extensive_data.add_step_data(model)
