from datetime import datetime, timedelta
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo
import pandas as pd
from simon.solver import Model
from marloes.handlers import (
    Handler,
    BatteryHandler,
    DemandHandler,
    GridHandler,
    SolarHandler,
    WindHandler,
)
from marloes.valley.env import EnergyValley

from test.util import get_new_config


class TestEnergyValleyEnv(unittest.TestCase):
    @patch("marloes.handlers.solar.read_series")
    @patch("marloes.handlers.demand.read_series")
    def setUp(self, mock_demand, mock_solar) -> None:
        # Reset id counter to make sure the other tests don't interfere
        Handler._id_counters = {}
        self.start_time = datetime(2025, 1, 1, tzinfo=ZoneInfo("UTC"))
        mock_series = pd.Series([100], index=[self.start_time])
        mock_demand.return_value = mock_series
        mock_solar.return_value = mock_series
        self.env = EnergyValley(get_new_config(), "Priorities")
        self.demand_handler = self.env.handlers[0]
        self.solar_handler = self.env.handlers[1]
        self.battery_handler = self.env.handlers[2]
        self.second_demand_handler = self.env.handlers[3]
        self.grid_handler = self.env.grid

    def test_handler_initialization(self):
        self.assertEqual(len(self.env.handlers), 4)  # 4 handlers
        # check if the handlers are of the right type
        self.assertIsInstance(self.demand_handler, DemandHandler)
        self.assertIsInstance(self.solar_handler, SolarHandler)
        self.assertIsInstance(self.battery_handler, BatteryHandler)
        self.assertIsInstance(self.second_demand_handler, DemandHandler)
        self.assertIsInstance(self.grid_handler, GridHandler)
        # check start time of each handler, should be equal to each other
        self.assertEqual(
            self.demand_handler.asset.state.time, self.solar_handler.asset.state.time
        )
        self.assertEqual(
            self.demand_handler.asset.state.time, self.battery_handler.asset.state.time
        )
        self.assertEqual(
            self.demand_handler.asset.state.time, self.grid_handler.asset.state.time
        )

    def test_handler_configurations(self):
        """
        Test the configurations of the handlers
        """
        # Demand
        self.assertEqual(self.demand_handler.id, "DemandHandler 0")
        self.assertEqual(self.demand_handler.asset.name, "Demand 0")
        self.assertEqual(self.demand_handler.asset.max_power_in, 150)
        # Solar
        self.assertEqual(self.solar_handler.asset.name, "Solar 0")
        self.assertEqual(self.solar_handler.id, "SolarHandler 0")
        self.assertEqual(self.solar_handler.asset.max_power_out, 900)
        # Battery
        self.assertEqual(self.battery_handler.asset.name, "Battery 0")
        self.assertEqual(self.battery_handler.asset.energy_capacity, 1000)

        # Check id of the second demand handler
        self.assertEqual(self.second_demand_handler.id, "DemandHandler 1")

    def test_grid_configuration(self):
        """
        Separate test for the grid configuration
        """
        self.assertEqual(self.grid_handler.asset.name, "Grid_One")
        self.assertEqual(self.grid_handler.asset.max_power_in, 1000)
        self.assertEqual(self.grid_handler.asset.max_power_out, float("inf"))

    def test__get_targets(self):
        """
        Test the _get_targets method
        """
        # Test the targets of the demand handler (should be empty)
        self.assertEqual(
            self.env._get_targets(self.demand_handler),
            [],
        )
        # Test the targets of the solar handler (should have all demand, battery/electrolyser and grid handlers)
        self.assertEqual(
            self.env._get_targets(self.solar_handler) + [(self.grid_handler.asset, 1)],
            [
                (self.demand_handler.asset, 3),
                (self.battery_handler.asset, 2),
                (self.second_demand_handler.asset, 3),
                (self.grid_handler.asset, 1),
            ],
        )
        # Test the targets of the battery handler (should have demand and grid handlers)
        self.assertEqual(
            self.env._get_targets(self.battery_handler)
            + [(self.grid_handler.asset, 1)],
            [
                (self.demand_handler.asset, 3),
                (self.second_demand_handler.asset, 3),
                (self.grid_handler.asset, 1),
            ],
        )
        # Test the targets of the grid handler (should be able to supply demand and flexible assets)
        self.assertEqual(
            self.env._get_targets(self.grid_handler),
            [
                (self.demand_handler.asset, -1),
                (self.battery_handler.asset, -1),
                (self.second_demand_handler.asset, -1),
            ],
        )

    def test_model_initialization(self):
        """
        Test if the model is initialized correctly
        """
        self.assertIsInstance(self.env.model, Model)
        num_nodes = len(self.env.model.graph.nodes)
        num_handlers = len(self.env.handlers)
        self.assertEqual(
            num_nodes, num_handlers + 2
        )  # each handler should be a node (add grid)
        # edges should be supply targets thus solar (5) + battery (3) + grid (4)
        self.assertEqual(len(self.env.model.graph.edges), 11)
        # check if the handlers are in the model
        for handler in self.env.handlers:
            self.assertIn(handler.asset, self.env.model.graph.nodes)
        # check if the connections are in the model
        for handler in self.env.handlers:
            for target, _ in self.env._get_targets(handler):
                self.assertIn((handler.asset, target), self.env.model.graph.edges)

    def test_step(self):
        """
        Test the step method
        """
        # dummy actions
        actions = {self.demand_handler.id: 0, self.battery_handler.id: 0}
        observation, reward, done, info = self.env.step(
            actions=actions, normalize=False
        )
        # and if the state time is updated with self.env.time_step
        self.assertEqual(
            self.demand_handler.asset.state.time,
            self.start_time + timedelta(seconds=self.env.time_step),
        )

    def test_algorithm_targets(self):
        """
        Test the priority mapping.
        """
        # SolarHandler targets in non-priorities mode
        solar_targets = self.env._get_targets(self.solar_handler)
        self.assertEqual(
            solar_targets + [(self.grid_handler.asset, -1)],
            [
                (self.demand_handler.asset, 3),
                (self.battery_handler.asset, 2),
                (self.second_demand_handler.asset, 3),
                (self.grid_handler.asset, -1),
            ],
        )

        # BatteryHandler targets in non-priorities mode
        battery_targets = self.env._get_targets(self.battery_handler)
        self.assertEqual(
            battery_targets + [(self.grid_handler.asset, -1)],
            [
                (self.demand_handler.asset, 3),
                (self.second_demand_handler.asset, 3),
                (self.grid_handler.asset, -1),
            ],
        )

        # GridHandler targets in non-priorities mode
        grid_targets = self.env._get_targets(self.grid_handler)
        self.assertEqual(
            grid_targets,
            [
                (self.demand_handler.asset, -1),
                (self.battery_handler.asset, -1),
                (self.second_demand_handler.asset, -1),
            ],
        )


class TestEnergyValleyEnvWithTwoBatteries(unittest.TestCase):
    @patch("marloes.handlers.solar.read_series")
    @patch("marloes.handlers.demand.read_series")
    def setUp(self, mock_demand, mock_solar) -> None:
        # Reset id counter to make sure the other tests don't interfere
        Handler._id_counters = {}
        self.start_time = datetime(2025, 1, 1, tzinfo=ZoneInfo("UTC"))
        mock_series = pd.Series([100], index=[self.start_time])
        mock_demand.return_value = mock_series
        mock_solar.return_value = mock_series
        config = get_new_config()
        config["handlers"].append(
            {
                "type": "battery",
                "efficiency": 0.9,
                "power": 100,
                "energy_capacity": 1000,
            }
        )
        self.env = EnergyValley(config, "Priorities")
        self.demand_handler = self.env.handlers[0]
        self.solar_handler = self.env.handlers[1]
        self.battery_handler = self.env.handlers[2]
        self.second_demand_handler = self.env.handlers[3]
        self.second_battery_handler = self.env.handlers[4]
        self.grid_handler = self.env.grid

    def test_handler_initialization(self):
        self.assertEqual(len(self.env.handlers), 5)  # 5 handlers
        # check if the handlers are of the right type
        self.assertIsInstance(self.demand_handler, DemandHandler)
        self.assertIsInstance(self.solar_handler, SolarHandler)
        self.assertIsInstance(self.battery_handler, BatteryHandler)
        self.assertIsInstance(self.second_demand_handler, DemandHandler)
        self.assertIsInstance(self.second_battery_handler, BatteryHandler)
        self.assertIsInstance(self.grid_handler, GridHandler)
        # check start time of each handler, should be equal to each other
        self.assertEqual(
            self.demand_handler.asset.state.time, self.solar_handler.asset.state.time
        )
        self.assertEqual(
            self.demand_handler.asset.state.time, self.battery_handler.asset.state.time
        )
        self.assertEqual(
            self.demand_handler.asset.state.time, self.grid_handler.asset.state.time
        )

    def test_battery_connections(self):
        """
        Test the connections of the second battery handler
        """
        self.assertEqual(
            self.env._get_targets(self.battery_handler)
            + [(self.grid_handler.asset, 1)],
            [
                (self.demand_handler.asset, 3),
                (self.second_demand_handler.asset, 3),
                (self.second_battery_handler.asset, -2),
                (self.grid_handler.asset, 1),
            ],
        )
        self.assertEqual(
            self.env._get_targets(self.second_battery_handler)
            + [(self.grid_handler.asset, 1)],
            [
                (self.demand_handler.asset, 3),
                (self.battery_handler.asset, -2),
                (self.second_demand_handler.asset, 3),
                (self.grid_handler.asset, 1),
            ],
        )


class TestEnergyValleyEnvWithWind(unittest.TestCase):
    @patch("marloes.handlers.wind.read_series")
    @patch("marloes.handlers.solar.read_series")
    @patch("marloes.handlers.demand.read_series")
    def setUp(self, mock_demand, mock_solar, mock_wind) -> None:
        # Reset id counter to make sure the other tests don't interfere
        Handler._id_counters = {}
        self.start_time = datetime(2025, 1, 1, tzinfo=ZoneInfo("UTC"))
        mock_series = pd.Series([100], index=[self.start_time])
        mock_demand.return_value = mock_series + 10
        mock_solar.return_value = mock_series - 10
        mock_wind.return_value = mock_series - 20
        config = get_new_config()
        config["handlers"].append(
            {
                "type": "wind",
                "location": "Onshore",
                "power": 1400,
                "AC": 1200,
                "curtailable_by_solver": True,
            }
        )
        self.env = EnergyValley(config, "Priorities")
        self.demand_handler = self.env.handlers[0]
        self.solar_handler = self.env.handlers[1]
        self.battery_handler = self.env.handlers[2]
        self.demand_handler_2 = self.env.handlers[3]
        self.wind_handler = self.env.handlers[4]
        self.grid = self.env.grid

    def test_handler_initialization(self):
        self.assertEqual(len(self.env.handlers), 5)
        self.assertIsInstance(self.demand_handler, DemandHandler)
        self.assertIsInstance(self.solar_handler, SolarHandler)
        self.assertIsInstance(self.battery_handler, BatteryHandler)
        self.assertIsInstance(self.demand_handler_2, DemandHandler)
        self.assertIsInstance(self.wind_handler, WindHandler)
        self.assertIsInstance(self.grid, GridHandler)
        # check start time of each handler, should be equal to each other
        self.assertEqual(
            self.demand_handler.asset.state.time, self.solar_handler.asset.state.time
        )
        self.assertEqual(
            self.demand_handler.asset.state.time, self.battery_handler.asset.state.time
        )
        self.assertEqual(
            self.demand_handler.asset.state.time, self.wind_handler.asset.state.time
        )
        self.assertEqual(
            self.demand_handler.asset.state.time, self.grid.asset.state.time
        )

    def test_wind_connections(self):
        """
        Test the connections of the wind handler
        """
        self.assertEqual(
            self.env._get_targets(self.wind_handler) + [(self.grid.asset, -1)],
            [
                (self.demand_handler.asset, 3),
                (self.battery_handler.asset, 2),
                (self.demand_handler_2.asset, 3),
                (self.grid.asset, -1),
            ],
        )
