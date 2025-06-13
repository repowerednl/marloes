import unittest
from unittest.mock import patch, MagicMock
from freezegun import freeze_time
from datetime import datetime, timedelta
from simon.assets.battery import Battery
from simon.data.battery_data import BatteryState
from marloes.handlers.electrolyser import ElectrolyserHandler


CONFIG = {
    "name": "ElectrolyserOne",
    "power": 50.0,
    "max_power_in": 100.0,
    "max_power_out": 40.0,
    "max_state_of_charge": 1.0,
    "min_state_of_charge": 0.0,
    "energy_capacity": 100.0,
    "ramp_up_rate": 0.2,
    "ramp_down_rate": 0.2,
    "input_efficiency": 0.5,
    "output_efficiency": 0.5,
    "degradation_function": None,
}


@freeze_time("2023-01-01 12:00:00")
class TestElectrolyserHandler(unittest.TestCase):
    def setUp(self) -> None:
        self.electrolyser_handler = ElectrolyserHandler(
            start_time=datetime.now(), config=CONFIG
        )

    def test_init(self):
        self.assertIsInstance(self.electrolyser_handler.asset, Battery)
        self.assertEqual(self.electrolyser_handler.asset.name, "ElectrolyserOne")
        self.assertEqual(self.electrolyser_handler.asset.max_power_in, 50.0)
        self.assertEqual(
            self.electrolyser_handler.asset.max_power_out, 40.0
        )  # Enforced
        self.assertEqual(self.electrolyser_handler.asset.max_state_of_charge, 1.0)
        self.assertEqual(self.electrolyser_handler.asset.min_state_of_charge, 0.0)
        self.assertEqual(
            self.electrolyser_handler.asset.energy_capacity,
            100.0,
        )
        self.assertEqual(self.electrolyser_handler.asset.ramp_up_rate, 0.2)
        self.assertEqual(self.electrolyser_handler.asset.ramp_down_rate, 0.2)
        self.assertEqual(self.electrolyser_handler.asset.input_efficiency, 0.5)
        self.assertEqual(self.electrolyser_handler.asset.output_efficiency, 0.5)
        self.assertIsNone(self.electrolyser_handler.asset.degradation_function)
        # test the initial (default) state
        self.assertEqual(self.electrolyser_handler.asset.state.time, datetime.now())
        self.assertEqual(self.electrolyser_handler.asset.state.power, 0.0)
        self.assertEqual(self.electrolyser_handler.asset.state.state_of_charge, 0.5)
        self.assertEqual(self.electrolyser_handler.asset.state.degradation, 0.0)

    def test_partial_init(self):
        partial_config = {
            "power": 50.0,
            "energy_capacity": 66.0,
        }
        other_electrolyser_handler = ElectrolyserHandler(
            start_time=datetime.now(), config=partial_config
        )
        self.assertIsInstance(other_electrolyser_handler.asset, Battery)
        self.assertEqual(other_electrolyser_handler.asset.max_power_in, 50.0)
        self.assertEqual(other_electrolyser_handler.asset.max_power_out, 50.0)
        self.assertEqual(other_electrolyser_handler.asset.max_state_of_charge, 0.95)
        self.assertEqual(other_electrolyser_handler.asset.min_state_of_charge, 0.05)
        self.assertEqual(
            other_electrolyser_handler.asset.energy_capacity,
            66.0,
        )
        self.assertEqual(other_electrolyser_handler.asset.ramp_up_rate, 0.4)
        self.assertEqual(other_electrolyser_handler.asset.ramp_down_rate, 0.4)
        self.assertIsNotNone(other_electrolyser_handler.asset.degradation_function)
        # test the initial (default) state
        self.assertEqual(other_electrolyser_handler.asset.state.time, datetime.now())
        self.assertEqual(other_electrolyser_handler.asset.state.power, 0.0)
        self.assertEqual(other_electrolyser_handler.asset.state.state_of_charge, 0.5)
        self.assertEqual(other_electrolyser_handler.asset.state.degradation, 0.0)

    def test_action_mapping(self):
        self.assertEqual(self.electrolyser_handler.map_action_to_setpoint(-1.0), -50.0)
        self.assertEqual(self.electrolyser_handler.internal_clock, 0)
        self.assertEqual(self.electrolyser_handler.map_action_to_setpoint(0.0), 0.0)
        self.assertEqual(self.electrolyser_handler.internal_clock, 0)
        # only the 5th positive setpoint will be registered
        # do it 5 times, print every result
        for i in range(5):
            if i < 5:
                self.assertEqual(
                    self.electrolyser_handler.map_action_to_setpoint(1.0), 0.0
                )
            else:
                self.assertEqual(
                    self.electrolyser_handler.map_action_to_setpoint(1.0), 40.0
                )

    def test_get_state(self):
        """
        Tests whether the state without time or is_fcr is returned correctly.
        """
        state = self.electrolyser_handler.get_state(0)
        self.assertIn("power", state)
        self.assertIn("state_of_charge", state)
        self.assertIn("degradation", state)
        self.assertNotIn("time", state)
        self.assertNotIn("is_fcr", state)
        self.assertEqual(len(state), 3)

    def test__loss_discharge(self):
        """
        Test the loss of energy is SOC over time: loss is 1 - (1 / (2 * 24 * 3600))
        """
        # set the initial state
        initial_state = BatteryState(
            time=datetime.now(), power=20.0, state_of_charge=0.5, degradation=0.5
        )
        self.electrolyser_handler.asset.set_state(initial_state)
        self.electrolyser_handler._loss_discharge()
        expected_soc = 0.5 * (1 - (1 / (2 * 24 * 3600)))
        # make sure only the SOC is changed
        self.assertEqual(self.electrolyser_handler.asset.state.time, datetime.now())
        self.assertEqual(self.electrolyser_handler.asset.state.power, 20.0)
        self.assertEqual(
            self.electrolyser_handler.asset.state.state_of_charge, expected_soc
        )
        self.assertEqual(self.electrolyser_handler.asset.state.degradation, 0.5)

    # General handler test
    @patch("simon.assets.battery.Battery.set_setpoint")
    def test_act_sets_correct_setpoint(self, mock_set_setpoint):
        # Arrange
        timestamp = datetime.now()
        action = 0.5
        expected_value = 20.0

        # set_setpoint should return the expected value
        mock_set_setpoint.return_value = expected_value
        self.electrolyser_handler.act(action, timestamp)

        mock_set_setpoint.assert_called_once()
        setpoint_arg = mock_set_setpoint.call_args[0][0]

        # Verify the setpoint argument
        self.assertEqual(setpoint_arg.value, 0)
        self.assertEqual(setpoint_arg.start, timestamp)
        self.assertEqual(setpoint_arg.stop, timestamp + timedelta(minutes=1))
        self.assertEqual(setpoint_arg.type, "power")

        action = -0.5
        expected_value = -20.0

        # set_setpoint should return the expected value
        mock_set_setpoint.return_value = expected_value
        self.electrolyser_handler.act(action, timestamp)

        # Verify the setpoint argument
        self.assertEqual(setpoint_arg.value, 0)
        self.assertEqual(setpoint_arg.start, timestamp)
        self.assertEqual(setpoint_arg.stop, timestamp + timedelta(minutes=1))
        self.assertEqual(setpoint_arg.type, "power")
