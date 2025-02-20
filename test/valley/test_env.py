from datetime import datetime, timedelta
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo
import pandas as pd
from simon.solver import Model
from marloes.agents import (
    Agent,
    BatteryAgent,
    DemandAgent,
    GridAgent,
    SolarAgent,
    WindAgent,
)
from marloes.valley.env import EnergyValley

from test.util import get_new_config


class TestEnergyValleyEnv(unittest.TestCase):
    @patch("marloes.agents.solar.read_series")
    @patch("marloes.agents.demand.read_series")
    def setUp(self, mock_demand, mock_solar) -> None:
        # Reset id counter to make sure the other tests don't interfere
        Agent._id_counters = {}
        self.start_time = datetime(2025, 1, 1, tzinfo=ZoneInfo("UTC"))
        mock_series = pd.Series([100], index=[self.start_time])
        mock_demand.return_value = mock_series
        mock_solar.return_value = mock_series
        self.env = EnergyValley(get_new_config(), "Priorities")
        self.demand_agent = self.env.agents[0]
        self.solar_agent = self.env.agents[1]
        self.battery_agent = self.env.agents[2]
        self.second_demand_agent = self.env.agents[3]
        self.grid_agent = self.env.grid

    def test_agent_initialization(self):
        self.assertEqual(len(self.env.agents), 4)  # 4 agents
        # check if the agents are of the right type
        self.assertIsInstance(self.demand_agent, DemandAgent)
        self.assertIsInstance(self.solar_agent, SolarAgent)
        self.assertIsInstance(self.battery_agent, BatteryAgent)
        self.assertIsInstance(self.second_demand_agent, DemandAgent)
        self.assertIsInstance(self.grid_agent, GridAgent)
        # check start time of each agent, should be equal to each other
        self.assertEqual(
            self.demand_agent.asset.state.time, self.solar_agent.asset.state.time
        )
        self.assertEqual(
            self.demand_agent.asset.state.time, self.battery_agent.asset.state.time
        )
        self.assertEqual(
            self.demand_agent.asset.state.time, self.grid_agent.asset.state.time
        )

    def test_agent_configurations(self):
        """
        Test the configurations of the agents
        """
        # Demand
        self.assertEqual(self.demand_agent.id, "DemandAgent 0")
        self.assertEqual(self.demand_agent.asset.name, "Demand 0")
        self.assertEqual(self.demand_agent.asset.max_power_in, float("inf"))
        # Solar
        self.assertEqual(self.solar_agent.asset.name, "Solar 0")
        self.assertEqual(self.solar_agent.id, "SolarAgent 0")
        self.assertEqual(self.solar_agent.asset.max_power_out, 900)
        # Battery
        self.assertEqual(self.battery_agent.asset.name, "Battery 0")
        self.assertEqual(self.battery_agent.asset.energy_capacity, 1000)

        # Check id of the second demand agent
        self.assertEqual(self.second_demand_agent.id, "DemandAgent 1")

    def test_grid_configuration(self):
        """
        Separate test for the grid configuration
        """
        self.assertEqual(self.grid_agent.asset.name, "Grid_One")
        self.assertEqual(self.grid_agent.asset.max_power_in, 1000)
        self.assertEqual(self.grid_agent.asset.max_power_out, float("inf"))

    def test__get_targets(self):
        """
        Test the _get_targets method
        """
        # Test the targets of the demand agent (should be empty)
        self.assertEqual(
            self.env._get_targets(self.demand_agent),
            [],
        )
        # Test the targets of the solar agent (should have all demand, battery/electrolyser and grid agents)
        self.assertEqual(
            self.env._get_targets(self.solar_agent) + [(self.grid_agent.asset, 1)],
            [
                (self.demand_agent.asset, 3),
                (self.battery_agent.asset, 2),
                (self.second_demand_agent.asset, 3),
                (self.grid_agent.asset, 1),
            ],
        )
        # Test the targets of the battery agent (should have demand and grid agents)
        self.assertEqual(
            self.env._get_targets(self.battery_agent) + [(self.grid_agent.asset, 1)],
            [
                (self.demand_agent.asset, 3),
                (self.second_demand_agent.asset, 3),
                (self.grid_agent.asset, 1),
            ],
        )
        # Test the targets of the grid agent (should be able to supply demand and flexible assets)
        self.assertEqual(
            self.env._get_targets(self.grid_agent),
            [
                (self.demand_agent.asset, -1),
                (self.battery_agent.asset, -1),
                (self.second_demand_agent.asset, -1),
            ],
        )

    def test_model_initialization(self):
        """
        Test if the model is initialized correctly
        """
        self.assertIsInstance(self.env.model, Model)
        num_nodes = len(self.env.model.graph.nodes)
        num_agents = len(self.env.agents)
        self.assertEqual(
            num_nodes, num_agents + 2
        )  # each agent should be a node (add grid)
        # edges should be supply targets thus solar (5) + battery (3) + grid (4)
        self.assertEqual(len(self.env.model.graph.edges), 11)
        # check if the agents are in the model
        for agent in self.env.agents:
            self.assertIn(agent.asset, self.env.model.graph.nodes)
        # check if the connections are in the model
        for agent in self.env.agents:
            for target, _ in self.env._get_targets(agent):
                self.assertIn((agent.asset, target), self.env.model.graph.edges)

    def test_step(self):
        """
        Test the step method
        """
        # dummy actions
        actions = {self.demand_agent.id: 0, self.battery_agent.id: 0}
        observation, reward, done, info = self.env.step(actions=actions)
        # and if the state time is updated with self.env.time_step
        self.assertEqual(
            self.demand_agent.asset.state.time,
            self.start_time + timedelta(seconds=self.env.time_step),
        )

    def test_algorithm_targets(self):
        """
        Test the priority mapping.
        """
        # SolarAgent targets in non-priorities mode
        solar_targets = self.env._get_targets(self.solar_agent)
        self.assertEqual(
            solar_targets + [(self.grid_agent.asset, -1)],
            [
                (self.demand_agent.asset, 3),
                (self.battery_agent.asset, 2),
                (self.second_demand_agent.asset, 3),
                (self.grid_agent.asset, -1),
            ],
        )

        # BatteryAgent targets in non-priorities mode
        battery_targets = self.env._get_targets(self.battery_agent)
        self.assertEqual(
            battery_targets + [(self.grid_agent.asset, -1)],
            [
                (self.demand_agent.asset, 3),
                (self.second_demand_agent.asset, 3),
                (self.grid_agent.asset, -1),
            ],
        )

        # GridAgent targets in non-priorities mode
        grid_targets = self.env._get_targets(self.grid_agent)
        self.assertEqual(
            grid_targets,
            [
                (self.demand_agent.asset, -1),
                (self.battery_agent.asset, -1),
                (self.second_demand_agent.asset, -1),
            ],
        )


class TestEnergyValleyEnvWithTwoBatteries(unittest.TestCase):
    @patch("marloes.agents.solar.read_series")
    @patch("marloes.agents.demand.read_series")
    def setUp(self, mock_demand, mock_solar) -> None:
        # Reset id counter to make sure the other tests don't interfere
        Agent._id_counters = {}
        self.start_time = datetime(2025, 1, 1, tzinfo=ZoneInfo("UTC"))
        mock_series = pd.Series([100], index=[self.start_time])
        mock_demand.return_value = mock_series
        mock_solar.return_value = mock_series
        config = get_new_config()
        config["agents"].append(
            {
                "type": "battery",
                "efficiency": 0.9,
                "power": 100,
                "energy_capacity": 1000,
            }
        )
        self.env = EnergyValley(config, "Priorities")
        self.demand_agent = self.env.agents[0]
        self.solar_agent = self.env.agents[1]
        self.battery_agent = self.env.agents[2]
        self.second_demand_agent = self.env.agents[3]
        self.second_battery_agent = self.env.agents[4]
        self.grid_agent = self.env.grid

    def test_agent_initialization(self):
        self.assertEqual(len(self.env.agents), 5)  # 5 agents
        # check if the agents are of the right type
        self.assertIsInstance(self.demand_agent, DemandAgent)
        self.assertIsInstance(self.solar_agent, SolarAgent)
        self.assertIsInstance(self.battery_agent, BatteryAgent)
        self.assertIsInstance(self.second_demand_agent, DemandAgent)
        self.assertIsInstance(self.second_battery_agent, BatteryAgent)
        self.assertIsInstance(self.grid_agent, GridAgent)
        # check start time of each agent, should be equal to each other
        self.assertEqual(
            self.demand_agent.asset.state.time, self.solar_agent.asset.state.time
        )
        self.assertEqual(
            self.demand_agent.asset.state.time, self.battery_agent.asset.state.time
        )
        self.assertEqual(
            self.demand_agent.asset.state.time, self.grid_agent.asset.state.time
        )

    def test_battery_connections(self):
        """
        Test the connections of the second battery agent
        """
        self.assertEqual(
            self.env._get_targets(self.battery_agent) + [(self.grid_agent.asset, 1)],
            [
                (self.demand_agent.asset, 3),
                (self.second_demand_agent.asset, 3),
                (self.second_battery_agent.asset, -2),
                (self.grid_agent.asset, 1),
            ],
        )
        self.assertEqual(
            self.env._get_targets(self.second_battery_agent)
            + [(self.grid_agent.asset, 1)],
            [
                (self.demand_agent.asset, 3),
                (self.battery_agent.asset, -2),
                (self.second_demand_agent.asset, 3),
                (self.grid_agent.asset, 1),
            ],
        )


class TestEnergyValleyEnvWithWind(unittest.TestCase):
    @patch("marloes.agents.wind.read_series")
    @patch("marloes.agents.solar.read_series")
    @patch("marloes.agents.demand.read_series")
    def setUp(self, mock_demand, mock_solar, mock_wind) -> None:
        # Reset id counter to make sure the other tests don't interfere
        Agent._id_counters = {}
        self.start_time = datetime(2025, 1, 1, tzinfo=ZoneInfo("UTC"))
        mock_series = pd.Series([100], index=[self.start_time])
        mock_demand.return_value = mock_series + 10
        mock_solar.return_value = mock_series - 10
        mock_wind.return_value = mock_series - 20
        config = get_new_config()
        config["agents"].append(
            {
                "type": "wind",
                "location": "Onshore",
                "power": 1400,
                "AC": 1200,
                "curtailable_by_solver": True,
            }
        )
        self.env = EnergyValley(config, "Priorities")
        self.demand_agent = self.env.agents[0]
        self.solar_agent = self.env.agents[1]
        self.battery_agent = self.env.agents[2]
        self.demand_agent_2 = self.env.agents[3]
        self.wind_agent = self.env.agents[4]
        self.grid = self.env.grid

    def test_agent_initialization(self):
        self.assertEqual(len(self.env.agents), 5)
        self.assertIsInstance(self.demand_agent, DemandAgent)
        self.assertIsInstance(self.solar_agent, SolarAgent)
        self.assertIsInstance(self.battery_agent, BatteryAgent)
        self.assertIsInstance(self.demand_agent_2, DemandAgent)
        self.assertIsInstance(self.wind_agent, WindAgent)
        self.assertIsInstance(self.grid, GridAgent)
        # check start time of each agent, should be equal to each other
        self.assertEqual(
            self.demand_agent.asset.state.time, self.solar_agent.asset.state.time
        )
        self.assertEqual(
            self.demand_agent.asset.state.time, self.battery_agent.asset.state.time
        )
        self.assertEqual(
            self.demand_agent.asset.state.time, self.wind_agent.asset.state.time
        )
        self.assertEqual(self.demand_agent.asset.state.time, self.grid.asset.state.time)

    def test_wind_connections(self):
        """
        Test the connections of the wind agent
        """
        self.assertEqual(
            self.env._get_targets(self.wind_agent) + [(self.grid.asset, -1)],
            [
                (self.demand_agent.asset, 3),
                (self.battery_agent.asset, 2),
                (self.demand_agent_2.asset, 3),
                (self.grid.asset, -1),
            ],
        )
