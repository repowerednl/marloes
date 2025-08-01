import unittest
from freezegun import freeze_time
from datetime import datetime, timedelta
import numpy as np
import pytest
from simon.assets.demand import Demand
from simon.datasource.data_source import (
    DummyDataSource,
)
from marloes.agents.demand import DemandAgent

CONFIG = {
    "name": "DemandOne",
    "profile": "Farm",
    "scale": 1,
    "constant_demand": 5.0,
    "curtailable_by_solver": True,
    "upward_dispatchable": False,
}


@freeze_time("2023-01-01 12:00:00")
class TestDemandAgent(unittest.TestCase):
    @pytest.mark.slow
    def test_init(self):
        demand_agent = DemandAgent(start_time=datetime.now(), config=CONFIG)
        self.assertIsInstance(demand_agent.asset, Demand)
        self.assertEqual(demand_agent.asset.name, "DemandOne")
        self.assertIsInstance(demand_agent.asset.data_source, DummyDataSource)
        self.assertEqual(demand_agent.asset.data_source.value, 5.0)
        # also test the get_state() method
        state = demand_agent.get_state(0)
        # state should have forecast, nomination and nomination_fraction
        self.assertIn("forecast", state)
        self.assertIn("nomination", state)
        self.assertIn("nomination_fraction", state)

    @pytest.mark.slow
    def test_partial_init(self):
        partial_config = {
            "profile": "Farm",
        }
        demand_agent = DemandAgent(start_time=datetime.now(), config=partial_config)
        self.assertIsInstance(demand_agent.asset, Demand)
        self.assertFalse(demand_agent.asset.curtailable_by_solver)
