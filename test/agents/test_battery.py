import unittest
from freezegun import freeze_time
from datetime import datetime, timedelta
from simon.assets.battery import Battery
from simon.data.battery_data import BatteryState
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
    @freeze_time("2023-01-01 12:00:00")
    def setUp(self) -> None:
        self.battery_agent = BatteryAgent(CONFIG)
        self.initial_state = BatteryState(
            time=datetime.now(),
            power=-20,
            state_of_charge=0.5,
            degradation=0.0
        )
        self.battery_agent.model.set_state(self.initial_state)

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
