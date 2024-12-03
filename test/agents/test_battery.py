import unittest
from freezegun import freeze_time
from datetime import datetime, timedelta
from simon.assets.battery import Battery
from marloes.agents.battery import BatteryAgent

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
class TestBatteryAgent(unittest.TestCase):
    def setUp(self) -> None:
        self.battery_agent = BatteryAgent(start_time=datetime.now(), config=CONFIG)

    def test_init(self):
        self.assertIsInstance(self.battery_agent.asset, Battery)
        self.assertEqual(self.battery_agent.asset.name, "BatteryOne")
        self.assertEqual(self.battery_agent.asset.max_power_in, 50.0)
        self.assertEqual(self.battery_agent.asset.max_power_out, 40.0)  # Enforced
        self.assertEqual(self.battery_agent.asset.max_state_of_charge, 1.0)
        self.assertEqual(self.battery_agent.asset.min_state_of_charge, 0.0)
        self.assertEqual(self.battery_agent.asset.energy_capacity, 100.0)
        self.assertEqual(self.battery_agent.asset.ramp_up_rate, 10.0)
        self.assertEqual(self.battery_agent.asset.ramp_down_rate, 10.0)
        self.assertEqual(self.battery_agent.asset.input_efficiency, 0.9**0.5)
        self.assertEqual(self.battery_agent.asset.output_efficiency, 0.9**0.5)
        self.assertIsNone(self.battery_agent.asset.degradation_function)
        # test the initial (default) state
        self.assertEqual(self.battery_agent.asset.state.time, datetime.now())
        self.assertEqual(self.battery_agent.asset.state.power, 0.0)
        self.assertEqual(self.battery_agent.asset.state.state_of_charge, 0.5)
        self.assertEqual(self.battery_agent.asset.state.degradation, 0.0)

    def test_partial_init(self):
        partial_config = {
            "power": 50.0,
            "energy_capacity": 100.0,
        }
        self.battery_agent = BatteryAgent(
            start_time=datetime.now(), config=partial_config
        )
        self.assertIsInstance(self.battery_agent.asset, Battery)
        self.assertEqual(self.battery_agent.asset.name, "Battery")
        self.assertEqual(self.battery_agent.asset.max_power_in, 50.0)
        self.assertEqual(self.battery_agent.asset.max_power_out, 50.0)
        self.assertEqual(self.battery_agent.asset.max_state_of_charge, 0.95)
        self.assertEqual(self.battery_agent.asset.min_state_of_charge, 0.05)
        self.assertEqual(self.battery_agent.asset.energy_capacity, 100.0)
        self.assertEqual(self.battery_agent.asset.ramp_up_rate, 50.0)
        self.assertEqual(self.battery_agent.asset.ramp_down_rate, 50.0)
        self.assertIsNotNone(self.battery_agent.asset.degradation_function)
        # test the initial (default) state
        self.assertEqual(self.battery_agent.asset.state.time, datetime.now())
        self.assertEqual(self.battery_agent.asset.state.power, 0.0)
        self.assertEqual(self.battery_agent.asset.state.state_of_charge, 0.5)
        self.assertEqual(self.battery_agent.asset.state.degradation, 0.0)
