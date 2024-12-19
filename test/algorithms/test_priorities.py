import unittest
from unittest.mock import patch

import pandas as pd

from marloes.algorithms.base import AlgorithmType
from marloes.algorithms.priorities import Priorities


def get_new_config() -> dict:
    return {
        "algorithm": "priorities",
        "epochs": 10,
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
                "power": 100,
            },
        ],
        # no grid agent should default to name="Grid", max_power_in and max_power_out should be inf
    }


class TestPriorities(unittest.TestCase):
    @patch("marloes.agents.solar.read_series", return_value=pd.Series())
    @patch("marloes.agents.demand.read_series", return_value=pd.Series())
    @patch("simon.assets.supply.Supply.load_default_state")
    @patch("simon.assets.demand.Demand.load_default_state")
    def setUp(self, *mocks) -> None:
        with patch("marloes.results.saver.Saver._save_config_to_yaml"), patch(
            "marloes.results.saver.Saver._update_simulation_number", return_value=0
        ), patch("marloes.results.saver.Saver._validate_folder"):
            self.alg = Priorities(config=get_new_config())

    def test_init(self):
        # no saving
        self.assertEqual(self.alg.epochs, 10)
        self.assertEqual(len(self.alg.environment.agents), 3)
        self.assertEqual(self.alg.algorithm_type, AlgorithmType.PRIORITIES)

    def test_agent_types(self):
        # check if the agents are of the right type
        self.assertEqual(len(self.alg.environment.agents), 3)
        self.assertEqual(
            [agent.__class__.__name__ for agent in self.alg.environment.agents],
            ["DemandAgent", "SolarAgent", "BatteryAgent"],
        )
        self.assertEqual(self.alg.environment.grid.asset.name, "Grid")

    def test_grid(self):
        # check if the grid agent is correctly initialized (default)
        grid = self.alg.environment.grid
        self.assertEqual(grid.asset.name, "Grid")
        self.assertEqual(grid.asset.max_power_in, float("inf"))
        self.assertEqual(grid.asset.max_power_out, float("inf"))

    @patch("marloes.valley.env.EnergyValley.step")
    @patch("marloes.valley.env.EnergyValley.reset")
    def test_train(self, mock_reset, mock_step):
        mock_reset.return_value = {}, {}
        mock_step.return_value = (1, 2, 3, 4)
        self.alg.train()
        self.assertEqual(mock_reset.call_count, 1)
        self.assertEqual(mock_step.call_count, self.alg.epochs)
