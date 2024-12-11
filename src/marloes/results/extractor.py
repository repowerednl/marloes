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

    def __init__(self, chunk_size: int = 0):
        """
        Initialize the Extractor with preallocated numpy arrays for metrics.
        """
        self.chunk_size = chunk_size
        self.i = 0
        self.start_time = time.time()
        size = MINUTES_IN_A_YEAR // self.chunk_size

        # Marl(oes) info
        self.elapsed_time = np.zeros(size)
        self.loss = np.zeros(size)
        self.action_probability_distribution = np.zeros(size)

        # Metrics/Reward info
        self.grid_to_demand = np.zeros(size)
        self.demand_state = np.zeros(size)
        self.grid_state = np.zeros(size)

        # Emission info
        self.total_solar_production = np.zeros(size)
        self.total_battery_production = np.zeros(size)
        self.total_wind_production = np.zeros(size)
        self.total_grid_production = np.zeros(size)

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
        if self.i >= len(self.elapsed_time):
            raise IndexError("Extractor has reached its maximum capacity.")

        # Tracking
        self.elapsed_time[self.i] = time.time() - self.start_time

        # Marl(oes) info
        # TODO: Implement loss and action_probability_distribution tracking
        # self.loss[self.i] = loss
        # self.action_probability_distribution[self.i] = action_probability_distribution

        # Metrics/Reward info
        self.grid_to_demand[self.i] = self._get_total_flow_between_types(
            model, type1=Connection, type2=Demand
        )
        power_data, output_power_data = self.get_current_power_by_type(model)
        self.demand_state[self.i] = power_data.get(DemandAgent.__name__, 0.0)
        self.grid_state[self.i] = power_data.get(GridAgent.__name__, 0.0)

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

    def from_files(self, uid: int):
        """
        Extract information from files based on a unique identifier.
        """
        # TODO: Implement file-based extraction logic
        self.update()

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
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Aggregate the current power production/intake for all assets, grouped by their type.
        """
        power_data = defaultdict(float)
        output_power_data = defaultdict(float)

        for asset in model.graph.nodes:
            power = asset.state.power
            asset_type = asset.name.split()[0]  # Take generic part of the name

            power_data[asset_type] += power
            output_power_data[asset_type] += max(0, power)

        return dict(power_data), dict(output_power_data)
