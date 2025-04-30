import unittest
from unittest.mock import patch
import pandas as pd
import pytest
from marloes.algorithms.priorities import Priorities
from marloes.agents.base import Agent

from test.util import get_accurate_observation, get_mock_observation


def get_new_config() -> dict:
    return {
        "algorithm": "priorities",
        "training_steps": 100,
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
        "replay_buffers": {
            "real_capacity": 1000,
            "model_capacity": 1000,
        },
    }


class TestPriorities(unittest.TestCase):
    @patch("marloes.agents.solar.read_series", return_value=pd.Series())
    @patch("marloes.agents.demand.read_series", return_value=pd.Series())
    @patch("simon.assets.supply.Supply.load_default_state")
    @patch("simon.assets.demand.Demand.load_default_state")
    def setUp(self, *mocks) -> None:
        with patch("marloes.results.saver.Saver._save_config_to_yaml"), patch(
            "marloes.results.saver.Saver._update_simulation_number", return_value=0
        ), patch("marloes.results.saver.Saver._validate_folder"), patch(
            "marloes.valley.env.EnergyValley._get_full_observation", return_value={}
        ), patch(
            "marloes.agents.base.Agent.get_state", return_value={}
        ), patch(
            "marloes.valley.env.Extractor.store_reward", return_value=None, create=True
        ):
            Agent._id_counters = {}
            self.alg = Priorities(config=get_new_config())

    def test_init(self):
        self.assertEqual(self.alg.training_steps, 100)
        self.assertEqual(len(self.alg.environment.agents), 3)

    def test_agent_types(self):
        self.assertEqual(len(self.alg.environment.agents), 3)
        self.assertEqual(
            [agent.__class__.__name__ for agent in self.alg.environment.agents],
            ["DemandAgent", "SolarAgent", "BatteryAgent"],
        )
        self.assertEqual(self.alg.environment.grid.asset.name, "Grid")

    def test_grid(self):
        grid = self.alg.environment.grid
        self.assertEqual(grid.asset.name, "Grid")
        self.assertEqual(grid.asset.max_power_in, float("inf"))
        self.assertEqual(grid.asset.max_power_out, float("inf"))

    @patch("marloes.valley.env.EnergyValley.step")
    @patch("marloes.valley.env.EnergyValley.reset")
    def test_train(self, mock_reset, mock_step):
        mock_reset.return_value = {}, {}
        mock_step.return_value = ({}, {}, {}, {})
        self.alg.train()
        self.assertEqual(mock_reset.call_count, 1)
        self.assertEqual(mock_step.call_count, self.alg.training_steps)

    def test__get_net_power(self):
        mock_obs = get_mock_observation(
            battery_soc=[0.5], solar_power=2.0, wind_power=1.0
        )
        net_power = self.alg._get_net_forecasted_power(mock_obs)
        self.assertEqual(net_power, 0.0)

        mock_obs = get_mock_observation(
            battery_soc=[0.5], solar_power=3.0, wind_power=1.0
        )
        net_power = self.alg._get_net_forecasted_power(mock_obs)
        self.assertEqual(net_power, 3.0)

    def test__determine_battery_actions_charge(self):
        obs = get_mock_observation(battery_soc=[0.5], solar_power=3.0, wind_power=1.0)
        batteries = self.alg._get_batteries(obs)
        actions = self.alg._determine_battery_actions(
            net_power=3.0, batteries=batteries
        )
        self.assertEqual(
            actions["BatteryAgent 0"],
            self.alg.environment.agents[2].asset.energy_capacity,
        )

        battery_config = {
            "type": "battery",
            "energy_capacity": 500,
            "efficiency": 0.9,
            "power": 100,
        }
        self.alg.environment._add_agent(battery_config)
        obs = get_mock_observation(
            battery_soc=[0.5, 0.6], solar_power=3.0, wind_power=1.0
        )
        batteries = self.alg._get_batteries(obs)
        actions = self.alg._determine_battery_actions(
            net_power=3.0, batteries=batteries
        )
        self.assertEqual(actions["BatteryAgent 0"], 500)
        self.assertEqual(actions["BatteryAgent 1"], 250)

    def test__determine_battery_actions_discharge(self):
        obs = get_mock_observation(battery_soc=[0.5], solar_power=1.0, wind_power=1.0)
        batteries = self.alg._get_batteries(obs)
        actions = self.alg._determine_battery_actions(
            net_power=-3.0, batteries=batteries
        )
        self.assertEqual(
            actions["BatteryAgent 0"],
            -self.alg.environment.agents[2].asset.energy_capacity,
        )

        battery_config = {
            "type": "battery",
            "energy_capacity": 500,
            "efficiency": 0.9,
            "power": 100,
        }
        self.alg.environment._add_agent(battery_config)
        obs = get_mock_observation(
            battery_soc=[0.5, 0.6], solar_power=1.0, wind_power=1.0
        )
        batteries = self.alg._get_batteries(obs)
        actions = self.alg._determine_battery_actions(
            net_power=-3.0, batteries=batteries
        )
        self.assertEqual(actions["BatteryAgent 0"], -500)
        self.assertEqual(actions["BatteryAgent 1"], -250)


@pytest.mark.slow
class TestPrioritiesSlow(unittest.TestCase):
    def setUp(self) -> None:
        Agent._id_counters = {}
        with patch(
            "marloes.results.saver.Saver._update_simulation_number", return_value=0
        ):
            self.alg = Priorities(config=get_new_config())

    def test_get_actions(self):
        observations = get_accurate_observation(self.alg)
        self.assertNotIn("forecast", observations["BatteryAgent 0"])
        self.assertIn("forecast", observations["SolarAgent 0"])
        self.assertIn("forecast", observations["DemandAgent 0"])

        actions = self.alg.get_actions(observations)
        self.assertIn("BatteryAgent 0", actions)

        mock_obs = get_mock_observation(
            battery_soc=[0.5], solar_power=3.0, wind_power=1.0
        )
        actions = self.alg.get_actions(mock_obs)
        self.assertEqual(
            actions["BatteryAgent 0"],
            self.alg.environment.agents[2].asset.energy_capacity,
        )
