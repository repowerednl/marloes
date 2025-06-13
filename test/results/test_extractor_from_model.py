# tests/test_extractor.py

import unittest
from collections import defaultdict

from marloes.factories import (
    BatteryFactory,
    ConnectionFactory,
    DemandFactory,
    SolarFactory,
)
from simon.solver import Model

from marloes.results.extractor import Extractor

MINUTES_IN_A_YEAR = 525600


class TestExtractorFromModel(unittest.TestCase):
    def setUp(self):
        """
        Set up a Model instance with a single Connection, Supply, and BatteryHandler using factories.
        """
        # Initialize the Model
        self.model = Model()

        # Create mock assets using factories
        self.grid = ConnectionFactory()
        self.solar = SolarFactory()
        self.solar_2 = SolarFactory()
        self.battery = BatteryFactory()
        self.demand = DemandFactory()

        # Add assets to the model with target connections and PrioFlow
        self.model.add_asset(
            self.grid, targets=[(self.battery, 1.0), (self.demand, 1.0)]
        )
        self.model.add_asset(self.solar, targets=[(self.battery, 0.5)])
        self.model.add_asset(self.solar_2, targets=[(self.battery, 0.5)])

        # Manually set flows in the model's edge_flow_tracker to simulate a timestep
        self.model.edge_flow_tracker = {
            (self.grid, self.battery): 80.0,
            (self.grid, self.demand): 70.0,
            (self.solar, self.battery): 30.0,
            (self.solar_2, self.battery): 25.0,
        }

        # Update asset_flow_tracker based on edge flows
        self.model.asset_flow_tracker = defaultdict(float)
        for (source, target), flow in self.model.edge_flow_tracker.items():
            self.model.asset_flow_tracker[source] += flow
            self.model.asset_flow_tracker[target] -= flow

        # Initialize the Extractor
        self.extractor = Extractor(chunk_size=1000)

    def test_from_model_power_aggregation(self):
        """
        Test that Extractor.from_model correctly aggregates power by asset type.
        """
        # Assign specific power values to assets
        self.solar.state.power = 100.0
        self.solar_2.state.power = 25.0
        self.grid.state.power = 200.0
        self.battery.state.power = -50.0
        self.demand.state.power = 0.0

        self.extractor.from_model(self.model)

        expected_power_data = {
            "SolarHandler": 125.0,
            "GridHandler": 200.0,
            "BatteryHandler": 0.0,
            "DemandHandler": 0.0,
        }

        actual_power_data = self.extractor.get_current_power_by_type(self.model)

        self.assertDictEqual(
            actual_power_data,
            expected_power_data,
            f"Expected power data {expected_power_data}, got {actual_power_data}",
        )

    def test_from_model_no_flows(self):
        """
        Test that Extractor.from_model handles models with no flows correctly.
        """
        # Reset the flow trackers to simulate no flows
        self.model.edge_flow_tracker = {}
        self.model.asset_flow_tracker = defaultdict(float)
        self.extractor.from_model(self.model)

        # Expect all extracted metrics to remain at their default values (0.0)
        self.assertEqual(self.extractor.total_solar_production[0], 0.0)
        self.assertEqual(self.extractor.total_battery_production[0], 0.0)
        self.assertEqual(self.extractor.total_grid_production[0], 0.0)

    def test_from_model_max_capacity(self):
        """
        Test that Extractor.from_model raises an IndexError when maximum capacity is reached.
        """
        # Set the index to the last valid position
        self.extractor.i = len(self.extractor.elapsed_time) - 1

        try:
            self.extractor.from_model(self.model)
            # update the model
            self.extractor.update()
            # and add additional information
            self.extractor.add_additional_info_from_model(self.model)
        except IndexError:
            self.fail("from_model raised IndexError unexpectedly when at last index")

        # Increment index beyond capacity and expect IndexError
        with self.assertRaises(IndexError):
            self.extractor.from_model(self.model)
            # update the model
            self.extractor.update()
            # and add additional information
            self.extractor.add_additional_info_from_model(self.model)
