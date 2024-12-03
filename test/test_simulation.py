import unittest
from unittest.mock import patch
from freezegun import freeze_time

import pandas as pd

from marloes.simulation import Simulation
from marloes.algorithms.base import AlgorithmType


def get_new_config():  # function to return a new configuration, pop caused issues
    return {
        "algorithm": AlgorithmType.MODEL_BASED,
        "epochs": 100,
        "agents": [
            {
                "type": "demand",
                "scale": 1.5,
                "profile": "Farm",
            },
            {
                "type": "solar",
                "AC": 900,
                "DC": 1000,
                "orientation": "EW",
            },
            {
                "type": "battery",
                "energy_capacity": 1000,
                "efficiency": 0.9,
                "max_power_in": 100,
                "max_power_out": 100,
            },
        ],
        # no grid agent should default to name="Grid", max_power_in and max_power_out should be inf
    }


@freeze_time("2023-01-01 12:00:00")
class TestSimulation(unittest.TestCase):
    def setUp(self) -> None:
        with patch("marloes.data.util.read_series", return_value=pd.Series()):
            self.sim = Simulation(config=get_new_config())
            self.sim_saving = Simulation(
                config=get_new_config(), save_energy_flows=True
            )

    def test_init(self):
        # no saving
        self.assertEqual(self.sim.epochs, 100)
        self.assertEqual(len(self.sim.valley.agents), 4)
        self.assertFalse(self.sim.saving)
        self.assertIsNone(self.sim.flows, None)
        self.assertEqual(self.sim.algorithm, AlgorithmType.MODEL_BASED)
        # saving
        self.assertTrue(self.sim_saving.saving)
        self.assertIsInstance(self.sim_saving.flows, list)

    def test_agent_types(self):
        # check if the agents are of the right type
        self.assertEqual(len(self.sim.valley.agents), 4)
        self.assertEqual(len(self.sim_saving.valley.agents), 4)
        self.assertEqual(
            [agent.__class__.__name__ for agent in self.sim.valley.agents],
            ["DemandAgent", "SolarAgent", "BatteryAgent", "GridAgent"],
        )
        self.assertEqual(
            [agent.__class__.__name__ for agent in self.sim_saving.valley.agents],
            ["DemandAgent", "SolarAgent", "BatteryAgent", "GridAgent"],
        )

    def test_grid(self):
        # check if the grid agent is correctly initialized (default)
        grid = self.sim.valley.agents[-1]
        self.assertEqual(grid.asset.name, "Grid")
        self.assertEqual(grid.asset.max_power_in, float("inf"))
        self.assertEqual(grid.asset.max_power_out, float("inf"))
