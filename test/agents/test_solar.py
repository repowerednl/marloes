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
from marloes.handlers.demand import DemandHandler
from marloes.handlers.solar import SolarHandler


@freeze_time("2023-01-01 12:00:00")
class TestSolarHandler(unittest.TestCase):
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
        solar_handler = SolarHandler(start_time=datetime.now(), config=config)
        self.assertIsInstance(solar_handler.asset, Supply)
        self.assertEqual(solar_handler.asset.name, "SolarOne")
        self.assertEqual(solar_handler.asset.max_power_out, 800)
        self.assertFalse(solar_handler.asset.curtailable_by_solver)
        self.assertFalse(solar_handler.asset.upward_dispatchable)

    @pytest.mark.slow
    @patch("simon.assets.supply.Supply.load_default_state")
    def test_partial_init(self, mock_default_state):
        partial_config = {
            "orientation": "S",
            "DC": 1000,
            "AC": 800,
        }
        solar_handler = SolarHandler(start_time=datetime.now(), config=partial_config)
        self.assertIsInstance(solar_handler.asset, Supply)
        self.assertEqual(solar_handler.asset.max_power_out, 800)
        self.assertTrue(solar_handler.asset.curtailable_by_solver)
        self.assertFalse(solar_handler.asset.upward_dispatchable)

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
        solar_handler = SolarHandler(start_time=datetime.now(), config=config)
        self.assertEqual(solar_handler.map_action_to_setpoint(-1), 0)
        self.assertEqual(solar_handler.map_action_to_setpoint(0), 0)
        self.assertEqual(solar_handler.map_action_to_setpoint(0.5), 400)


class TestSolarHandlerGetState(unittest.TestCase):
    @patch.object(
        SolarHandler, "__init__", lambda self, *args, **kwargs: None
    )  # Bypass init
    def test_get_state_with_forecast(self):
        """Tests whether get_state correctly returns the forecast slice."""
        solar_handler = SolarHandler()
        solar_handler.forecast = np.linspace(0, 1, num=2000, dtype=np.float32)
        solar_handler.nominated_volume = [1, 2, 3, 4, 5]
        solar_handler.horizon = 1440

        # Mock asset state
        solar_handler.asset = MagicMock()
        solar_handler.asset.state.model_dump.return_value = {"power": 0.1}

        # Check initial state
        state = solar_handler.get_state(start_idx=0)
        self.assertIn("forecast", state)
        self.assertEqual(len(state["forecast"]), 1440)
        np.testing.assert_array_equal(state["forecast"], solar_handler.forecast[:1440])
        self.assertEqual(state["power"], 0.1)

        # Check nomination_fraction (exists and value should be 0.0 + power / nominated_volume)
        self.assertIn("nomination_fraction", state)
        self.assertEqual(state["nomination_fraction"], 0.0 + 0.1 / 1 / 60)

        # Check state at a later index
        state = solar_handler.get_state(start_idx=500)
        np.testing.assert_array_equal(
            state["forecast"], solar_handler.forecast[500 : 500 + 1440]
        )

    @patch.object(
        SolarHandler, "__init__", lambda self, *args, **kwargs: None
    )  # Bypass init
    def test_get_state_at_end_of_forecast(self):
        """Tests whether get_state correctly handles cases where the horizon exceeds the forecast."""
        solar_handler = SolarHandler()
        solar_handler.forecast = np.linspace(0, 1, num=1500, dtype=np.float32)
        solar_handler.nominated_volume = [1, 2, 3, 4, 5]
        solar_handler.horizon = 1440

        # Mock asset state
        solar_handler.asset = MagicMock()
        solar_handler.asset.state.model_dump.return_value = {"power": 0.5}

        # Test when start index is near the end (in the last hour: nominated volume = 5)
        start_idx = 1400
        state = solar_handler.get_state(start_idx=start_idx)

        # Forecast should only contain remaining values
        self.assertEqual(len(state["forecast"]), 100)
        np.testing.assert_array_equal(
            state["forecast"], solar_handler.forecast[1400:1500]
        )
        self.assertEqual(state["power"], 0.5)
        self.assertEqual(state["nomination_fraction"], 0.0 + 0.5 / 5 / 60)

        # when we call get_state with start_idx += 1, the nomination fraction should be updated with the updated power
        solar_handler.asset.state.model_dump.return_value = {"power": 0.2}
        state = solar_handler.get_state(start_idx=start_idx + 1)
        # nomination_fraction = 0.5 / 5 / 60 + 0.2 / 5 / 60
        self.assertEqual(state["nomination_fraction"], 0.5 / 5 / 60 + 0.2 / 5 / 60)

    @patch.object(
        SolarHandler, "__init__", lambda self, *args, **kwargs: None
    )  # Bypass init
    def test_get_state(self):
        """
        Tests whether the state without time is returned correctly.
        """
        solar_handler = SolarHandler()
        solar_handler.forecast = [1, 2, 3, 4, 5]
        solar_handler.nominated_volume = [1, 2, 3, 4, 5]
        solar_handler.horizon = 2
        # Mock asset state
        solar_handler.asset = MagicMock()
        solar_handler.asset.state.model_dump.return_value = {
            "time": datetime.now(),
            "power": 0.0,
            "available_power": 0.0,
        }

        state = solar_handler.get_state(0)
        self.assertIn("power", state)
        self.assertIn("available_power", state)
        self.assertIn("forecast", state)
        self.assertIn("nomination", state)
        self.assertNotIn("time", state)
        self.assertEqual(len(state), 5)
