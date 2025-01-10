import unittest
from unittest.mock import patch
from freezegun import freeze_time
from datetime import datetime
import numpy as np
import pytest
from simon.assets.supply import Supply
from simon.datasource.data_source import (
    DummyDataSource,
)
from marloes.agents.demand import DemandAgent
from marloes.agents.solar import SolarAgent


@freeze_time("2023-01-01 12:00:00")
class TestSolarAgent(unittest.TestCase):
    @pytest.mark.slow
    @patch("simon.assets.supply.Supply.load_default_state")
    def test_init(self, mock_default_state):
        config = {
            "name": "SolarOne",
            "orientation": "EW",
            "DC": 1000,
            "AC": 800,
            "curtailable_by_solver": False,
        }
        solar_agent = SolarAgent(start_time=datetime.now(), config=config)
        self.assertIsInstance(solar_agent.asset, Supply)
        self.assertEqual(solar_agent.asset.name, "SolarOne")
        self.assertEqual(solar_agent.asset.max_power_out, 800)
        self.assertFalse(solar_agent.asset.curtailable_by_solver)
        self.assertFalse(solar_agent.asset.upward_dispatchable)

    @pytest.mark.slow
    @patch("simon.assets.supply.Supply.load_default_state")
    def test_partial_init(self, mock_default_state):
        partial_config = {
            "orientation": "S",
            "DC": 1000,
            "AC": 800,
        }
        solar_agent = SolarAgent(start_time=datetime.now(), config=partial_config)
        self.assertIsInstance(solar_agent.asset, Supply)
        self.assertEqual(solar_agent.asset.max_power_out, 800)
        self.assertTrue(solar_agent.asset.curtailable_by_solver)
        self.assertFalse(solar_agent.asset.upward_dispatchable)

    @pytest.mark.slow
    @patch("simon.assets.supply.Supply.load_default_state")
    def test_action_mapping(self, mock_default_state):
        config = {
            "name": "SolarOne",
            "orientation": "EW",
            "DC": 1000,
            "AC": 800,
            "curtailable_by_solver": False,
        }
        solar_agent = SolarAgent(start_time=datetime.now(), config=config)
        self.assertEqual(solar_agent.map_action_to_setpoint(-1), 0)
        self.assertEqual(solar_agent.map_action_to_setpoint(0), 0)
        self.assertEqual(solar_agent.map_action_to_setpoint(0.5), 400)
