import unittest
from unittest.mock import MagicMock, patch
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


class TestSolarAgentGetState(unittest.TestCase):
    @patch.object(
        SolarAgent, "__init__", lambda self, *args, **kwargs: None
    )  # Bypass init
    def test_get_state_with_forecast(self):
        """Tests whether get_state correctly returns the forecast slice."""
        solar_agent = SolarAgent()
        solar_agent.forecast = np.linspace(0, 1, num=2000, dtype=np.float32)
        solar_agent.nominated_volume = [1, 2, 3, 4, 5]
        solar_agent.horizon = 1440

        # Mock asset state
        solar_agent.asset = MagicMock()
        solar_agent.asset.max_power_out = 5
        solar_agent.asset.state.model_dump.return_value = {"mocked_state": True}

        # Check initial state
        state = solar_agent.get_state(start_idx=0)
        self.assertIn("forecast", state)
        self.assertEqual(
            len(state["forecast"]), 4
        )  # mean, std, average_slope, turning_points
        used_forecast = solar_agent.forecast[:1440]
        mean = np.mean(used_forecast) / solar_agent.asset.max_power_out
        std = np.std(used_forecast) / solar_agent.asset.max_power_out
        average_slope = (
            np.mean(np.diff(used_forecast)) / solar_agent.asset.max_power_out
        )
        turning_points = np.sum(np.diff(np.sign(np.diff(used_forecast))) != 0) / len(
            used_forecast
        )
        np.testing.assert_array_equal(
            state["forecast"], np.array([mean, std, average_slope, turning_points])
        )
        self.assertEqual(state["mocked_state"], True)

        # Check state at a later index
        state = solar_agent.get_state(start_idx=500)
        used_forecast = solar_agent.forecast[500 : 500 + 1440]
        mean = np.mean(used_forecast) / solar_agent.asset.max_power_out
        std = np.std(used_forecast) / solar_agent.asset.max_power_out
        average_slope = (
            np.mean(np.diff(used_forecast)) / solar_agent.asset.max_power_out
        )
        turning_points = np.sum(np.diff(np.sign(np.diff(used_forecast))) != 0) / len(
            used_forecast
        )
        np.testing.assert_array_equal(
            state["forecast"], np.array([mean, std, average_slope, turning_points])
        )

    @patch.object(
        SolarAgent, "__init__", lambda self, *args, **kwargs: None
    )  # Bypass init
    def test_get_state_at_end_of_forecast(self):
        """Tests whether get_state correctly handles cases where the horizon exceeds the forecast."""
        solar_agent = SolarAgent()
        solar_agent.forecast = np.linspace(0, 1, num=1500, dtype=np.float32)
        solar_agent.nominated_volume = [1, 2, 3, 4, 5]
        solar_agent.horizon = 1440

        # Mock asset state
        solar_agent.asset = MagicMock()
        solar_agent.asset.max_power_out = 5
        solar_agent.asset.state.model_dump.return_value = {"mocked_state": True}

        # Test when start index is near the end
        start_idx = 1400
        state = solar_agent.get_state(start_idx=start_idx)

        # Forecast should only contain remaining values
        self.assertEqual(
            len(state["forecast"]), 4
        )  # mean, std, average_slope, turning_points
        used_forecast = solar_agent.forecast[1400:1500]
        mean = np.mean(used_forecast) / solar_agent.asset.max_power_out
        std = np.std(used_forecast) / solar_agent.asset.max_power_out
        average_slope = (
            np.mean(np.diff(used_forecast)) / solar_agent.asset.max_power_out
        )
        turning_points = np.sum(np.diff(np.sign(np.diff(used_forecast))) != 0) / len(
            used_forecast
        )
        np.testing.assert_array_equal(
            state["forecast"], np.array([mean, std, average_slope, turning_points])
        )
        self.assertEqual(state["mocked_state"], True)

    @patch.object(
        SolarAgent, "__init__", lambda self, *args, **kwargs: None
    )  # Bypass init
    def test_get_state_with_complex_forecast(self):
        """
        Since we transform the forecast into a set of features, we need to test whether this transformation is done correctly.
        """
        solar_agent = SolarAgent()
        solar_agent.forecast = np.array([2, 2, 4, 6, 8, 6, 4, 8, 6, 10])
        # forecast with mean 5.6/10, std about 2.5, average slope (sum differences / #differences (=10-1)) = 8/9, and 4 /10 = 2.5 turning points
        solar_agent.nominated_volume = [1, 2, 3, 4, 5]
        solar_agent.horizon = 10

        # Mock asset state
        solar_agent.asset = MagicMock()
        solar_agent.asset.max_power_out = 10
        solar_agent.asset.state.model_dump.return_value = {"mocked_state": True}

        # Check initial state
        state = solar_agent.get_state(start_idx=0)
        self.assertIn("forecast", state)
        self.assertEqual(len(state["forecast"]), 4)
        np.testing.assert_array_almost_equal(
            state["forecast"], np.array([0.56, 0.25, 0.089, 0.4]), decimal=3
        )

    @patch.object(
        SolarAgent, "__init__", lambda self, *args, **kwargs: None
    )  # Bypass init
    def test_get_state(self):
        """
        Tests whether the state without time is returned correctly.
        """
        solar_agent = SolarAgent()
        solar_agent.forecast = [1, 2, 3, 4, 5]
        solar_agent.nominated_volume = [1, 2, 3, 4, 5]
        solar_agent.horizon = 2
        # Mock asset state
        solar_agent.asset = MagicMock()
        solar_agent.asset.max_power_out = 5
        solar_agent.asset.state.model_dump.return_value = {
            "time": datetime.now(),
            "power": 0.0,
            "available_power": 0.0,
        }

        state = solar_agent.get_state(0)
        self.assertIn("power", state)
        self.assertIn("available_power", state)
        self.assertIn("forecast", state)
        self.assertIn("nomination", state)
        self.assertNotIn("time", state)
        self.assertEqual(len(state), 4)
