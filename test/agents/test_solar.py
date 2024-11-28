import unittest
from unittest.mock import patch
from freezegun import freeze_time
from datetime import datetime
import numpy as np
from simon.assets.supply import Supply
from simon.datasource.data_source import (
    DummyDataSource,
)
from marloes.agents.demand import DemandAgent
from marloes.agents.solar import SolarAgent

CONFIG = {
    "name": "SolarOne",
    "orientation": "EW",
    "DC": 1000,
    "AC": 800,
    "curtailable_by_solver": False,
}


@freeze_time("2023-01-01 12:00:00")
class TestSolarAgent(unittest.TestCase):
    @patch("simon.assets.supply.Supply.load_default_state")
    def test_init(self, mock_default_state):
        solar_agent = SolarAgent(start_time=datetime.now(), config=CONFIG)
        self.assertIsInstance(solar_agent.asset, Supply)
        self.assertEqual(solar_agent.asset.name, "SolarOne")
        self.assertEqual(solar_agent.asset.max_power_out, np.inf)
        self.assertFalse(solar_agent.asset.curtailable_by_solver)
        self.assertFalse(solar_agent.asset.upward_dispatchable)

    @patch("simon.assets.supply.Supply.load_default_state")
    def test_partial_init(self, mock_default_state):
        partial_config = {
            "orientation": "S",
            "DC": 1000,
            "AC": 800,
        }
        solar_agent = SolarAgent(start_time=datetime.now(), config=partial_config)
        self.assertIsInstance(solar_agent.asset, Supply)
        self.assertEqual(solar_agent.asset.name, "Solar")
        self.assertEqual(solar_agent.asset.max_power_out, np.inf)
        self.assertTrue(solar_agent.asset.curtailable_by_solver)
        self.assertFalse(solar_agent.asset.upward_dispatchable)
