import unittest
from unittest.mock import patch
from freezegun import freeze_time
from datetime import datetime, timedelta
from simon.assets.electrolyser import Electrolyser
from marloes.agents.electrolyser import ElectrolyserAgent

CONFIG = {
    "name": "ElectrolyserOverwrite",
    "capacity": 1000,
    "conversion_efficiency": 64,
    "max_power_in": 50.0,
    "min_power_in": 0.5,
    "slew_rate_up": 0.1,
    "slew_rate_down": 0.1,
    "startup_time": 10,
    "anode_pressure": 1,
    "storage_pressure": 1,
}


@freeze_time("2023-01-01 12:00:00")
class TestElectrolyserAgent(unittest.TestCase):
    def setUp(self) -> None:
        self.electrolyser_agent = ElectrolyserAgent(
            start_time=datetime.now(), config=CONFIG
        )

    def test_init(self):
        self.assertIsInstance(self.electrolyser_agent.asset, Electrolyser)
        self.assertEqual(self.electrolyser_agent.asset.name, "ElectrolyserOverwrite")
        self.assertEqual(self.electrolyser_agent.asset.max_power_in, 50.0)
        self.assertEqual(self.electrolyser_agent.asset.max_power_out, 10)  # Enforced
        self.assertEqual(self.electrolyser_agent.asset.conversion_efficiency, 0.64)
        self.assertEqual(self.electrolyser_agent.asset.slew_rate_up, 0.1)
        self.assertEqual(self.electrolyser_agent.asset.slew_rate_down, 0.1)
        self.assertEqual(self.electrolyser_agent.asset.startup_time, 10)
        self.assertEqual(self.electrolyser_agent.asset.anode_pressure, 1)
        self.assertEqual(self.electrolyser_agent.asset.storage_pressure, 1)
        # test the initial (default) state
        self.assertEqual(self.electrolyser_agent.asset.state.time, datetime.now())
        self.assertEqual(self.electrolyser_agent.asset.state.power, 0.0)

    def test_partial_init(self):
        partial_config = {
            "capacity": 50.0,
            "conversion_efficiency": 0.64,
        }
        self.electrolyser_agent = ElectrolyserAgent(
            start_time=datetime.now(), config=partial_config
        )
        self.assertIsInstance(self.electrolyser_agent.asset, Electrolyser)
        self.assertEqual(self.electrolyser_agent.asset.name, "Electrolyser")
        self.assertEqual(self.electrolyser_agent.asset.max_power_in, 50.0)
        self.assertEqual(self.electrolyser_agent.asset.max_power_out, 50.0)
        self.assertEqual(self.electrolyser_agent.asset.conversion_efficiency, 0.64)
        self.assertEqual(self.electrolyser_agent.asset.slew_rate_up, 0.1)
        self.assertEqual(self.electrolyser_agent.asset.slew_rate_down, 0.1)
        self.assertEqual(self.electrolyser_agent.asset.startup_time, 0)
        self.assertEqual(self.electrolyser_agent.asset.anode_pressure, 1)
        self.assertEqual(self.electrolyser_agent.asset.storage_pressure, 1)
        # test the initial (default) state
        self.assertEqual(self.electrolyser_agent.asset.state.time, datetime.now())
        self.assertEqual(self.electrolyser_agent.asset.state.power, 0.0)

    def test_action_mapping(self):
        self.assertEqual(self.electrolyser_agent.map_action_to_setpoint(-1.0), 50.0)
        self.assertEqual(self.electrolyser_agent.map_action_to_setpoint(0.0), 0.0)
        self.assertEqual(self.electrolyser_agent.map_action_to_setpoint(1.0), 40.0)

    # General agent test
    @patch("simon.assets.electrolyser.Electrolyser.set_setpoint")
    def test_act_sets_correct_setpoint(self, mock_set_setpoint):
        # Arrange
        timestamp = datetime.now()
        action = 0.5
        expected_value = 20.0

        # set_setpoint should return the expected value
        mock_set_setpoint.return_value = expected_value

        self.electrolyser_agent.act(action, timestamp)

        mock_set_setpoint.assert_called_once()
        setpoint_arg = mock_set_setpoint.call_args[0][0]

        # Verify the setpoint argument
        self.assertEqual(setpoint_arg.value, expected_value)
        self.assertEqual(setpoint_arg.start, timestamp)
        self.assertEqual(setpoint_arg.stop, timestamp + timedelta(minutes=1))
        self.assertEqual(setpoint_arg.type, "power")
