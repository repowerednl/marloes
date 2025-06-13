import unittest
from unittest.mock import Mock, patch
from freezegun import freeze_time
from datetime import datetime, timedelta
from simon.assets.battery import Battery
from marloes.handlers.battery import BatteryHandler

CONFIG = {
    "name": "BatteryOne",
    "power": 50.0,
    "max_power_in": 100.0,
    "max_power_out": 40.0,
    "max_state_of_charge": 1.0,
    "min_state_of_charge": 0.0,
    "energy_capacity": 100.0,
    "ramp_up_rate": 10.0,
    "ramp_down_rate": 10.0,
    "efficiency": 0.9,
    "degradation_function": None,
}


@freeze_time("2023-01-01 12:00:00")
class TestBatteryHandler(unittest.TestCase):
    def setUp(self) -> None:
        self.battery_handler = BatteryHandler(start_time=datetime.now(), config=CONFIG)

    def test_init(self):
        self.assertIsInstance(self.battery_handler.asset, Battery)
        self.assertEqual(self.battery_handler.asset.name, "BatteryOne")
        self.assertEqual(self.battery_handler.asset.max_power_in, 50.0)
        self.assertEqual(self.battery_handler.asset.max_power_out, 40.0)  # Enforced
        self.assertEqual(self.battery_handler.asset.max_state_of_charge, 1.0)
        self.assertEqual(self.battery_handler.asset.min_state_of_charge, 0.0)
        self.assertEqual(self.battery_handler.asset.energy_capacity, 100.0)
        self.assertEqual(self.battery_handler.asset.ramp_up_rate, 10.0)
        self.assertEqual(self.battery_handler.asset.ramp_down_rate, 10.0)
        self.assertEqual(self.battery_handler.asset.input_efficiency, 0.9**0.5)
        self.assertEqual(self.battery_handler.asset.output_efficiency, 0.9**0.5)
        self.assertIsNone(self.battery_handler.asset.degradation_function)
        # test the initial (default) state
        self.assertEqual(self.battery_handler.asset.state.time, datetime.now())
        self.assertEqual(self.battery_handler.asset.state.power, 0.0)
        self.assertEqual(self.battery_handler.asset.state.state_of_charge, 0.5)
        self.assertEqual(self.battery_handler.asset.state.degradation, 0.0)

    def test_partial_init(self):
        partial_config = {
            "power": 50.0,
            "energy_capacity": 100.0,
        }
        self.battery_handler = BatteryHandler(
            start_time=datetime.now(), config=partial_config
        )
        self.assertIsInstance(self.battery_handler.asset, Battery)
        self.assertEqual(self.battery_handler.asset.max_power_in, 50.0)
        self.assertEqual(self.battery_handler.asset.max_power_out, 50.0)
        self.assertEqual(self.battery_handler.asset.max_state_of_charge, 0.95)
        self.assertEqual(self.battery_handler.asset.min_state_of_charge, 0.05)
        self.assertEqual(self.battery_handler.asset.energy_capacity, 100.0)
        self.assertEqual(self.battery_handler.asset.ramp_up_rate, 50.0)
        self.assertEqual(self.battery_handler.asset.ramp_down_rate, 50.0)
        self.assertIsNotNone(self.battery_handler.asset.degradation_function)
        # test the initial (default) state
        self.assertEqual(self.battery_handler.asset.state.time, datetime.now())
        self.assertEqual(self.battery_handler.asset.state.power, 0.0)
        self.assertEqual(self.battery_handler.asset.state.state_of_charge, 0.5)
        self.assertEqual(self.battery_handler.asset.state.degradation, 0.0)

    def test_action_mapping(self):
        self.assertEqual(self.battery_handler.map_action_to_setpoint(-1.0), -50.0)
        self.assertEqual(self.battery_handler.map_action_to_setpoint(0.0), 0.0)
        self.assertEqual(self.battery_handler.map_action_to_setpoint(1.0), 40.0)

    def test_get_state(self):
        """
        Tests whether the state without time or is_fcr is returned correctly.
        """
        state = self.battery_handler.get_state(0)
        self.assertIn("power", state)
        self.assertIn("state_of_charge", state)
        self.assertIn("degradation", state)
        self.assertNotIn("time", state)
        self.assertNotIn("is_fcr", state)
        self.assertEqual(len(state), 3)

    # General handler test
    @patch("simon.assets.battery.Battery.set_setpoint")
    def test_act_sets_correct_setpoint(self, mock_set_setpoint):
        # Arrange
        timestamp = datetime.now()
        action = 0.5
        expected_value = 20.0

        # set_setpoint should return the expected value
        mock_set_setpoint.return_value = expected_value

        self.battery_handler.act(action, timestamp)

        mock_set_setpoint.assert_called_once()
        setpoint_arg = mock_set_setpoint.call_args[0][0]

        # Verify the setpoint argument
        self.assertEqual(setpoint_arg.value, expected_value)
        self.assertEqual(setpoint_arg.start, timestamp)
        self.assertEqual(setpoint_arg.stop, timestamp + timedelta(minutes=1))
        self.assertEqual(setpoint_arg.type, "power")
