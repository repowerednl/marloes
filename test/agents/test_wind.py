import unittest
from unittest.mock import patch, MagicMock
from freezegun import freeze_time
from datetime import datetime
import numpy as np
import pytest
from simon.assets.supply import Supply
from marloes.agents.wind import WindAgent


@freeze_time("2023-01-01 12:00:00")
class TestWindAgent(unittest.TestCase):
    @pytest.mark.slow
    @patch("simon.assets.supply.Supply.load_default_state")
    def test_init(self, mock_default_state):
        config = {
            "name": "WindOne",
            "location": "Onshore",
            "power": 1400,
            "AC": 1200,
            "curtailable_by_solver": False,
        }
        wind_agent = WindAgent(start_time=datetime.now(), config=config)
        self.assertIsInstance(wind_agent.asset, Supply)
        self.assertEqual(wind_agent.asset.name, "WindOne")
        self.assertEqual(wind_agent.asset.max_power_out, 1200)
        self.assertFalse(wind_agent.asset.curtailable_by_solver)
        self.assertFalse(wind_agent.asset.upward_dispatchable)

    @pytest.mark.slow
    @patch("simon.assets.supply.Supply.load_default_state")
    def test_partial_init(self, mock_default_state):
        partial_config = {
            "location": "Offshore",
            "AC": 1200,
            "power": 1000,
        }
        wind_agent = WindAgent(start_time=datetime.now(), config=partial_config)
        self.assertIsInstance(wind_agent.asset, Supply)
        self.assertEqual(wind_agent.asset.max_power_out, 1000)
        self.assertTrue(wind_agent.asset.curtailable_by_solver)
        self.assertFalse(wind_agent.asset.upward_dispatchable)

    @pytest.mark.slow
    @patch("simon.assets.supply.Supply.load_default_state")
    def test_action_mapping(self, mock_default_state):
        config = {
            "name": "WindOne",
            "location": "Onshore",
            "power": 1400,
            "AC": 1200,
            "curtailable_by_solver": False,
        }
        wind_agent = WindAgent(start_time=datetime.now(), config=config)
        self.assertEqual(wind_agent.map_action_to_setpoint(-1), 0)
        self.assertEqual(wind_agent.map_action_to_setpoint(0), 0)
        self.assertEqual(wind_agent.map_action_to_setpoint(0.5), 600)
        self.assertEqual(wind_agent.map_action_to_setpoint(1), 1200)

    @pytest.mark.slow
    @patch("simon.assets.supply.Supply.load_default_state")
    def test_wrong_config(self, mock_default_state):
        config = {
            "name": "WindOne",
            "location": "Onshore",
            "AC": 1200,
            "curtailable_by_solver": False,
        }
        with self.assertRaises(KeyError):
            WindAgent(start_time=datetime.now(), config=config)

    @patch.object(
        WindAgent, "__init__", lambda self, *args, **kwargs: None
    )  # Bypass init
    def test_get_state(self):
        """
        Tests whether the state without time is returned correctly.
        """
        wind_agent = WindAgent()
        wind_agent.forecast = [1, 2, 3, 4, 5]
        wind_agent.horizon = 2
        # Mock asset state
        wind_agent.asset = MagicMock()
        wind_agent.asset.state.model_dump.return_value = {
            "time": datetime.now(),
            "power": 0.0,
            "available_power": 0.0,
        }

        state = wind_agent.get_state(0)
        self.assertIn("power", state)
        self.assertIn("available_power", state)
        self.assertIn("forecast", state)
        self.assertNotIn("time", state)
        self.assertEqual(len(state), 3)
