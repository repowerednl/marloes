import unittest
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

class TestBatteryAgent(unittest.TestCase):

    def test_init(self):
        battery_agent = BatteryAgent(CONFIG)
        self.assertIsInstance(battery_agent.model, Battery)
        self.assertEqual(battery_agent.model.name, "BatteryOne")
        self.assertEqual(battery_agent.model.max_power_in, 50.0)
        self.assertEqual(battery_agent.model.max_power_out, 50.0)
        self.assertEqual(battery_agent.model.max_state_of_charge, 1.0)
        self.assertEqual(battery_agent.model.min_state_of_charge, 0.0)
        self.assertEqual(battery_agent.model.energy_capacity, 100.0)
        self.assertEqual(battery_agent.model.ramp_up_rate, 10.0)
        self.assertEqual(battery_agent.model.ramp_down_rate, 10.0)
        self.assertEqual(battery_agent.model.input_efficiency, 0.95)
        self.assertEqual(battery_agent.model.output_efficiency, 0.95)
        self.assertIsNone(battery_agent.model.degradation_function)
