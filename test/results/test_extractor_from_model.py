# tests/test_extractor.py

import unittest
import unittest.mock as mock
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


class TestExtractorFromModelSingleConnection(unittest.TestCase):
    def setUp(self):
        """
        Set up a Model instance with a single Connection, Supply, and BatteryAgent using factories.
        """
        # Initialize the Model
        self.model = Model()

        # Create mock assets using factories
        self.grid = ConnectionFactory()
        self.solar = SolarFactory()
        self.battery = BatteryFactory()
        self.demand = DemandFactory()

        # Add assets to the model with target connections and priorities
        self.model.add_asset(
            self.grid, targets=[(self.battery, 1.0), (self.demand, 1.0)]
        )
        self.model.add_asset(self.solar, targets=[(self.battery, 0.5)])

        # Manually set flows in the model's edge_flow_tracker to simulate a timestep
        self.model.edge_flow_tracker = {
            (self.grid, self.battery): 80.0,
            (self.grid, self.demand): 70.0,
            (self.solar, self.battery): 30.0,
        }

        # Update asset_flow_tracker based on edge flows
        self.model.asset_flow_tracker = defaultdict(float)
        for (source, target), flow in self.model.edge_flow_tracker.items():
            self.model.asset_flow_tracker[source] += flow
            self.model.asset_flow_tracker[target] -= flow

        # Initialize the Extractor
        self.extractor = Extractor(chunk_size=1000)

    def test_from_model_flow_extraction(self):
        """
        Test that Extractor.from_model correctly extracts a certain flow from the model.
        """
        self.extractor.from_model(self.model)

        # Grid to demand flow should be 70.0
        expected_flow = 70.0

        # Assert that the Extractor has correctly captured the total flow
        actual_flow = self.extractor.grid_to_demand[0]
        self.assertEqual(
            actual_flow,
            expected_flow,
            f"Expected total flow to be {expected_flow}, got {actual_flow}",
        )

    def test_from_model_power_aggregation(self):
        """
        Test that Extractor.from_model correctly aggregates power by asset type.
        """
        # Assign specific power values to assets
        self.solar.state.power = 100.0
        self.grid.state.power = 200.0
        self.battery.state.power = -50.0
        self.demand.state.power = 0.0

        self.extractor.from_model(self.model)
        expected_power_data = {
            "SolarAgent": 100.0,
            "GridAgent": 200.0,
            "BatteryAgent": -50.0,
            "DemandAgent": 0.0,
        }

        # Actual power data
        power_data, output_power_data = self.extractor.get_current_power_by_type(
            self.model
        )

        # Verify power_data
        self.assertDictEqual(
            power_data,
            expected_power_data,
            f"Expected power data {expected_power_data}, got {power_data}",
        )

        # Expected output power data (only positive values)
        expected_output_power_data = {
            "GridAgent": 200.0,
            "SolarAgent": 100.0,
            "BatteryAgent": 0.0,
            "DemandAgent": 0.0,
        }

        self.assertDictEqual(
            output_power_data,
            expected_output_power_data,
            f"Expected output power data {expected_output_power_data}, got {output_power_data}",
        )

    def test_from_model_no_flows(self):
        """
        Test that Extractor.from_model handles models with no flows correctly.
        """
        # Reset the flow trackers to simulate no flows
        self.model.reset_flow_tracker()
        self.extractor.from_model(self.model)

        # Expected total flow is 0.0
        expected_flow = 0.0

        actual_flow = self.extractor.grid_to_demand[0]
        self.assertEqual(
            actual_flow,
            expected_flow,
            f"Expected total flow to be {expected_flow}, got {actual_flow}",
        )

    def test_from_model_max_capacity(self):
        """
        Test that Extractor.from_model raises an IndexError when maximum capacity is reached.
        """
        # Set the index to the last valid position
        self.extractor.i = len(self.extractor.elapsed_time) - 1

        try:
            self.extractor.from_model(self.model)
        except IndexError:
            self.fail("from_model raised IndexError unexpectedly when at last index")

        # Increment index beyond capacity and expect IndexError
        with self.assertRaises(IndexError):
            self.extractor.from_model(self.model)
