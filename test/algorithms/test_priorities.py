import unittest
from unittest.mock import patch
import pandas as pd
import pytest
from marloes.algorithms.priorities import Priorities
from marloes.agents.base import Agent
from marloes.agents.battery import BatteryAgent

from test.util import get_accurate_observation, get_mock_observation


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

    def test__get_net_power(self):
        """
        Test the _get_net_power method, which sums the forecasts of relevant assets
        in the Priorities algorithm.
        """
        # demand is [2,3,4], solar is [2.0,2.0,2.0], wind is [1.0,1.0,1.0]
        mock_obs = get_mock_observation(
            battery_soc=[0.5], solar_power=2.0, wind_power=1.0
        )
        # net power should be 0.0
        net_power = self.alg._get_net_forecasted_power(mock_obs)
        self.assertEqual(net_power, 0.0)
        # demand is [2,3,4], solar is [3.0,3.0,3.0], wind is [1.0,1.0,1.0]
        mock_obs = get_mock_observation(
            battery_soc=[0.5], solar_power=3.0, wind_power=1.0
        )
        # net power should be 3.0
        net_power = self.alg._get_net_forecasted_power(mock_obs)
        self.assertEqual(net_power, 3.0)

    def test__determine_battery_actions_charge(self):
        """
        Test the _determine_battery_actions method, which determines the actions for the batteries
        once they qualify in the Priorities algorithm when they should be charging (positive net power).
        """
        # only one battery should charge to full capacity, when qualified, and positive net power
        obs = get_mock_observation(battery_soc=[0.5], solar_power=3.0, wind_power=1.0)
        batteries = self.alg._get_batteries(obs)
        actions = self.alg._determine_battery_actions(
            net_power=3.0, batteries=batteries, soc_threshold=0.4
        )
        self.assertEqual(
            actions["BatteryAgent 0"],
            self.alg.environment.agents[2].asset.energy_capacity,
        )
        # two batteries should both charge to half capacity
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
            net_power=3.0, batteries=batteries, soc_threshold=0.4
        )
        self.assertEqual(actions["BatteryAgent 0"], 500)
        self.assertEqual(actions["BatteryAgent 1"], 250)

    def test__determine_battery_actions_discharge(self):
        """
        Test the _determine_battery_actions method, which determines the actions for the batteries
        once they qualify in the Priorities algorithm when they should be discharging (negative net power).
        """
        # only one battery should discharge to full capacity
        obs = get_mock_observation(battery_soc=[0.5], solar_power=1.0, wind_power=1.0)
        batteries = self.alg._get_batteries(obs)
        actions = self.alg._determine_battery_actions(
            net_power=-3.0, batteries=batteries, soc_threshold=0.4
        )
        self.assertEqual(
            actions["BatteryAgent 0"],
            -self.alg.environment.agents[2].asset.energy_capacity,
        )
        # two batteries should both discharge to half capacity, to test this we add one battery
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
            net_power=-3.0, batteries=batteries, soc_threshold=0.4
        )
        self.assertEqual(actions["BatteryAgent 0"], -500)
        self.assertEqual(actions["BatteryAgent 1"], -250)

    def test__determine_battery_actions_not_qualified(self):
        """
        The battery should have a certain SOC before it should be allowed to discharge.
        We always charge if net power is positive.
        """
        # add batteries before calling _determine_battery_actions
        battery_configs = [
            {
                "type": "battery",
                "energy_capacity": 500,
                "efficiency": 0.9,
                "power": 100,
            },
            {
                "type": "battery",
                "energy_capacity": 200,
                "efficiency": 0.9,
                "power": 100,
            },
            {
                "type": "battery",
                "energy_capacity": 400,
                "efficiency": 0.9,
                "power": 100,
            },
        ]
        for config in battery_configs:
            self.alg.environment._add_agent(config)

        test_cases = [
            ([0.2], -3.0, 0.4, {"BatteryAgent 0": 0}),
            ([0.2, 0.5], -3.0, 0.4, {"BatteryAgent 0": 0, "BatteryAgent 1": -500}),
            (
                [0.2, 0.5, 0.3],
                -3.0,
                0.4,
                {"BatteryAgent 0": 0, "BatteryAgent 1": -500, "BatteryAgent 2": 0},
            ),
            (
                [0.2, 0.5, 0.3, 0.6],
                -3.0,
                0.4,
                {
                    "BatteryAgent 0": 0,
                    "BatteryAgent 1": -250,
                    "BatteryAgent 2": 0,
                    "BatteryAgent 3": -200,
                },
            ),
        ]

        for soc, net_power, soc_threshold, expected_actions in test_cases:
            obs = get_mock_observation(battery_soc=soc, solar_power=1.0, wind_power=1.0)
            batteries = self.alg._get_batteries(obs)
            actions = self.alg._determine_battery_actions(
                net_power=net_power, batteries=batteries, soc_threshold=soc_threshold
            )
            for agent, action in expected_actions.items():
                self.assertEqual(actions[agent], action)


@pytest.mark.slow
class TestPrioritiesSlow(unittest.TestCase):
    def setUp(self) -> None:
        self.alg = Priorities(config=get_new_config())

    def test_get_actions(self):
        """
        Only the battery should receive setpoints in the Priorities algorithm.
        """
        observations = get_accurate_observation(self.alg)
        # the battery should not have a forecast, but solar and demand should
        self.assertNotIn("forecast", observations["BatteryAgent 0"])
        self.assertIn("forecast", observations["SolarAgent 0"])
        self.assertIn("forecast", observations["DemandAgent 0"])
        # the battery should have a setpoint
        actions = self.alg.get_actions(observations)
        self.assertIn("BatteryAgent 0", actions)
        # use mock observation to test the actions, this should result in positive net power > charging to full capacity
        mock_obs = get_mock_observation(
            battery_soc=[0.5], solar_power=3.0, wind_power=1.0
        )
        actions = self.alg.get_actions(mock_obs)
        self.assertEqual(
            actions["BatteryAgent 0"],
            self.alg.environment.agents[2].asset.energy_capacity,
        )
