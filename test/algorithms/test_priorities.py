import unittest
from unittest.mock import patch
import pandas as pd
import pytest
from marloes.algorithms.priorities import Priorities
from marloes.agents.base import Agent


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

    def tearDown(self):
        # reset the Agent.__id_counters to 0
        Agent._id_counters = {}
        return super().tearDown()

    def test_init(self):
        # no saving
        self.assertEqual(self.alg.epochs, 10)
        self.assertEqual(len(self.alg.environment.agents), 3)

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
        mock_step.return_value = ({}, 2, 3, 4)
        self.alg.train()
        self.assertEqual(mock_reset.call_count, 1)
        self.assertEqual(mock_step.call_count, self.alg.epochs)


@pytest.mark.slow
class TestPrioritiesSlow(unittest.TestCase):
    def setUp(self) -> None:
        self.alg = Priorities(config=get_new_config())

    def test_get_actions(self):
        """
        Only the battery should receive setpoints in the Priorities algorithm.
        Testing two cases: 1 battery, 1 solar and 3 batteries, 1 solar.
        """
        from test.util import get_accurate_observation

        observations = get_accurate_observation(self.alg)
        actions = self.alg.get_actions(observations)
        self.assertEqual(actions, {})
        observations = get_accurate_observation(self.alg)
        actions = self.alg.get_actions(observations)
        # assert False == True
