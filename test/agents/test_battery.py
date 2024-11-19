import unittest
from freezegun import freeze_time
from datetime import datetime, timedelta
from simon.assets.battery import Battery
from marloes.agents.battery import BatteryAgent

CONFIG = {
    "name": "BatteryOne",
    "max_power_in": 50.0,
    "max_power_out": 50.0,
    "max_state_of_charge": 1.0,
    "min_state_of_charge": 0.0,
    "energy_capacity": 100.0,
    "ramp_up_rate": 10.0,
    "ramp_down_rate": 10.0,
    "efficiency": 0.9,
    "degradation_function": None
}

@freeze_time("2023-01-01 12:00:00")
class TestBatteryAgent(unittest.TestCase):
    def setUp(self) -> None:
        self.battery_agent = BatteryAgent(start_time=datetime.now(),config=CONFIG)

    def test_init(self):
        self.assertIsInstance(self.battery_agent.model, Battery)
        self.assertEqual(self.battery_agent.model.name, "BatteryOne")
        self.assertEqual(self.battery_agent.model.max_power_in, 50.0)
        self.assertEqual(self.battery_agent.model.max_power_out, 50.0)
        self.assertEqual(self.battery_agent.model.max_state_of_charge, 1.0)
        self.assertEqual(self.battery_agent.model.min_state_of_charge, 0.0)
        self.assertEqual(self.battery_agent.model.energy_capacity, 100.0)
        self.assertEqual(self.battery_agent.model.ramp_up_rate, 10.0)
        self.assertEqual(self.battery_agent.model.ramp_down_rate, 10.0)
        self.assertEqual(self.battery_agent.model.input_efficiency, 0.9**0.5)
        self.assertEqual(self.battery_agent.model.output_efficiency, 0.9**0.5)
        self.assertIsNone(self.battery_agent.model.degradation_function)
        # test the initial (default) state
        self.assertEqual(self.battery_agent.model.state.time, datetime.now())
        self.assertEqual(self.battery_agent.model.state.power, 0.0)
        self.assertEqual(self.battery_agent.model.state.state_of_charge, 0.5)
        self.assertEqual(self.battery_agent.model.state.degradation, 0.0)

