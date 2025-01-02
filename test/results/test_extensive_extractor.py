import unittest
from collections import defaultdict
import os
import pandas as pd
import numpy as np
from unittest.mock import patch

from marloes.factories import (
    BatteryFactory,
    ConnectionFactory,
    DemandFactory,
    SolarFactory,
)
from simon.solver import Model
from marloes.results.extractor import ExtensiveExtractor
from simon.simulation import SimulationResults


class TestExtensiveExtractor(unittest.TestCase):
    def setUp(self):
        """
        Set up a Model instance with mock assets and an ExtensiveExtractor.
        """
        # Initialize the Model
        self.model = Model()

        # Create mock assets using factories
        self.grid = ConnectionFactory()
        self.solar = SolarFactory()
        self.solar_2 = SolarFactory()
        self.battery = BatteryFactory()
        self.demand = DemandFactory()

        # Add assets to the model with target connections and priorities
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

        # Initialize the ExtensiveExtractor
        self.extractor = ExtensiveExtractor(chunk_size=1000)

    def test_from_model_extraction(self):
        """
        Test that ExtensiveExtractor correctly extracts data from the model.
        """
        # Assign specific power values to assets
        self.solar.state.power = 100.0
        self.solar_2.state.power = 25.0
        self.grid.state.power = 200.0
        self.battery.state.power = -50.0
        self.demand.state.power = 0.0

        self.extractor.from_model(self.model)

        # Check aggregated power
        expected_power_data = {
            "SolarAgent": 125.0,
            "GridAgent": 200.0,
            "BatteryAgent": 0.0,
            "DemandAgent": 0.0,
        }

        actual_power_data = self.extractor.get_current_power_by_type(self.model)
        self.assertDictEqual(
            actual_power_data,
            expected_power_data,
            f"Expected power data {expected_power_data}, got {actual_power_data}",
        )

        # Check if SimulationResults were populated
        self.assertIsNotNone(self.extractor.results)
        self.assertEqual(
            len(self.extractor.results.states), len(self.model.graph.nodes)
        )
        self.assertEqual(
            len(self.extractor.results.flows), len(self.model.edge_flow_tracker)
        )

    def test_from_model_no_flows(self):
        """
        Test that ExtensiveExtractor handles models with no flows correctly.
        """
        # Reset the flow trackers to simulate no flows
        self.model.edge_flow_tracker = {}
        self.model.asset_flow_tracker = defaultdict(float)
        self.extractor.from_model(self.model)

        # Check that results are initialized but empty
        self.assertIsNotNone(self.extractor.results)
        for _, flows in self.extractor.results.flows.items():
            self.assertEqual(len(flows), 0)

    def test_reaching_capacity(self):
        """
        Test that ExtensiveExtractor handles reaching capacity correctly.
        """
        # Set the index to the last valid position
        self.extractor.i = self.extractor.size - 1

        # Should not raise an error on the last valid index
        try:
            self.extractor.from_model(self.model)
        except IndexError:
            self.fail("from_model raised IndexError unexpectedly when at last index")

        # Increment index beyond capacity and expect IndexError
        with self.assertRaises(IndexError):
            self.extractor.from_model(self.model)
