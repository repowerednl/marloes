from datetime import datetime, timedelta
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo
import pandas as pd
from simon.solver import Model
from marloes.agents.base import Agent
from marloes.agents.demand import DemandAgent
from marloes.agents.solar import SolarAgent
from marloes.agents.battery import BatteryAgent
from marloes.agents.grid import GridAgent
from marloes.algorithms.types import AlgorithmType
from marloes.valley.env import EnergyValley


def get_new_config():  # function to return a new configuration, pop caused issues
    return {
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
                "efficiency": 0.9,
                "power": 100,
                "energy_capacity": 1000,
            },
            {
                "type": "demand",
                "scale": 1.5,
                "profile": "Farm",
            },
        ],
        "grid": {
            "name": "Grid_One",
            "max_power_in": 1000,
        },
        "algorithm": "priorities",
    }


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
        self.env = EnergyValley(config=get_new_config())
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
        self.assertEqual(self.demand_agent.asset.name, "Demand")
        self.assertEqual(self.demand_agent.asset.max_power_in, float("inf"))
        # Solar
        self.assertEqual(self.solar_agent.asset.name, "Solar")
        self.assertEqual(self.solar_agent.id, "SolarAgent 0")
        self.assertEqual(self.solar_agent.asset.max_power_out, 900)
        # Battery
        self.assertEqual(self.battery_agent.asset.name, "Battery")
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
        algorithm_type = AlgorithmType.PRIORITIES
        self.assertEqual(
            self.env._get_targets(self.demand_agent, algorithm_type=algorithm_type),
            [],
        )
        # Test the targets of the solar agent (should have all demand, battery/electrolyser and grid agents)
        self.assertEqual(
            self.env._get_targets(self.solar_agent, algorithm_type)
            + [(self.grid_agent.asset, 1)],
            [
                (self.demand_agent.asset, 3),
                (self.battery_agent.asset, 2),
                (self.second_demand_agent.asset, 3),
                (self.grid_agent.asset, 1),
            ],
        )
        # Test the targets of the battery agent (should have demand and grid agents)
        self.assertEqual(
            self.env._get_targets(self.battery_agent, algorithm_type)
            + [(self.grid_agent.asset, 1)],
            [
                (self.demand_agent.asset, 3),
                (self.second_demand_agent.asset, 3),
                (self.grid_agent.asset, 1),
            ],
        )
        # Test the targets of the grid agent (should be able to supply demand and flexible assets)
        self.assertEqual(
            self.env._get_targets(self.grid_agent, algorithm_type),
            [
                (self.demand_agent.asset, 3),
                (self.battery_agent.asset, 2),
                (self.second_demand_agent.asset, 3),
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
            num_nodes, num_agents + 1
        )  # each agent should be a node (add grid)
        # edges should be supply targets thus solar (4) + battery (3) + grid (4)
        self.assertEqual(len(self.env.model.graph.edges), 10)
        # check if the agents are in the model
        for agent in self.env.agents:
            self.assertIn(agent.asset, self.env.model.graph.nodes)
        # check if the connections are in the model
        for agent in self.env.agents:
            for target, _ in self.env._get_targets(
                agent, algorithm_type=AlgorithmType.PRIORITIES
            ):
                self.assertIn((agent.asset, target), self.env.model.graph.edges)

    def test_step(self):
        """
        Test the step method
        """
        # dummy actions
        actions = [0, 0, 0]
        observation, reward, done, info = self.env.step(actions=actions)
        # and if the state time is updated with self.env.time_step
        self.assertEqual(
            self.demand_agent.asset.state.time,
            self.start_time + timedelta(seconds=self.env.time_step),
        )

    def test_non_priorities_algorithm_targets(self):
        """
        Test the priority mapping when the algorithm type is NOT 'PRIORITIES'.
        """
        algorithm_type = AlgorithmType.MODEL_FREE

        # SolarAgent targets in non-priorities mode
        solar_targets = self.env._get_targets(self.solar_agent, algorithm_type)
        self.assertEqual(
            solar_targets + [(self.grid_agent.asset, 10)],
            [
                (self.demand_agent.asset, 0),
                (self.battery_agent.asset, 0),
                (self.second_demand_agent.asset, 0),
                (self.grid_agent.asset, 10),
            ],
        )

        # BatteryAgent targets in non-priorities mode
        battery_targets = self.env._get_targets(self.battery_agent, algorithm_type)
        self.assertEqual(
            battery_targets + [(self.grid_agent.asset, 10)],
            [
                (self.demand_agent.asset, 0),
                (self.second_demand_agent.asset, 0),
                (self.grid_agent.asset, 10),
            ],
        )

        # GridAgent targets in non-priorities mode
        grid_targets = self.env._get_targets(self.grid_agent, algorithm_type)
        self.assertEqual(
            grid_targets,
            [
                (self.demand_agent.asset, 10),
                (self.battery_agent.asset, 10),
                (self.second_demand_agent.asset, 10),
            ],
        )

    def test_priorities_algorithm_targets(self):
        """
        Test the priority mapping when the algorithm type is 'PRIORITIES'.
        """
        algorithm_type = AlgorithmType.PRIORITIES

        # SolarAgent targets in PRIORITIES mode
        solar_targets = self.env._get_targets(self.solar_agent, algorithm_type)
        self.assertEqual(
            solar_targets + [(self.grid_agent.asset, -1)],
            [
                (self.demand_agent.asset, 3),
                (self.battery_agent.asset, 2),
                (self.second_demand_agent.asset, 3),
                (self.grid_agent.asset, -1),
            ],
        )

        # GridAgent targets in PRIORITIES mode
        grid_targets = self.env._get_targets(self.grid_agent, algorithm_type)
        self.assertEqual(
            grid_targets,
            [
                (self.demand_agent.asset, 3),
                (self.battery_agent.asset, 2),
                (self.second_demand_agent.asset, 3),
            ],
        )
