import os
import time
from collections import defaultdict
from typing import Dict, Tuple, Type

import numpy as np
from simon.assets.demand import Demand
from simon.assets.grid import Connection
from simon.solver import Model

from marloes.agents.battery import BatteryAgent
from marloes.agents.demand import DemandAgent
from marloes.agents.grid import GridAgent
from marloes.agents.solar import SolarAgent

MINUTES_IN_A_YEAR = 525600


class Extractor:
    """
    Extractor class to gather and store simulation metrics.
    """

    def __init__(self, from_model: bool = True, chunk_size: int = 0):
        """
        Initialize the Extractor with preallocated numpy arrays for metrics.
        """
        self.chunk_size = chunk_size
        self.i = 0
        self.start_time = time.time()
        self.size = MINUTES_IN_A_YEAR // self.chunk_size

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
        self.grid_state[self.i] = list(model.graph.nodes)[-1].state.power

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

    def from_files(self, uid: int | None = None, dir: str = "results") -> None:
        """
        Extract information from files based on a unique identifier.
        If no identifier is given, the latest identifier is used.
        """
        # If uid is None, extract latest uid from results/uid.txt
        if uid is None:
            with open(f"{dir}/uid.txt", "r") as f:
                uid = int(f.read().strip()) - 1

        # Loop over each folder in the ./results directory
        # If there is a file in a folder with the given uid (ends in "_uid.npy"), extract the data and save it to
        # an attribute corresponding to the folder name
        # If there is no file with the given uid just skip the folder
        for folder in os.listdir(dir):
            if os.path.isdir(folder):
                for file in os.listdir(folder):
                    if file.endswith(f"_{uid}.npy"):
                        self.__setattr__(folder, np.load(f"{dir}/{folder}/{file}"))

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
    def __init__(self, from_model: bool = True, chunk_size: int = 0):
        super.__init__(from_model, chunk_size)

        if from_model:
            # Additional marl info
            self.action_probability_distribution = np.zeros(self.size)

            # Complete flows/state dataframe
            pass
