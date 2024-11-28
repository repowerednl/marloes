import unittest
from unittest.mock import patch
from freezegun import freeze_time
import pandas as pd
from marloes.agents.demand import DemandAgent
from marloes.agents.solar import SolarAgent
from marloes.agents.battery import BatteryAgent
from marloes.valley.env import EnergyValley


def get_new_config():  # function to return a new configuration, pop caused issues
    return {
        "agents": [
            {
                "type": "demand",
                "scale": 1.5,
                "profile": "Farm",
            },
            {
                "type": "solar",
                "AC": 900,
                "DC": 1000,
                "orientation": "EW",
            },
            {
                "type": "battery",
                "efficiency": 0.9,
                "max_power_in": 100,
                "max_power_out": 100,
                "energy_capacity": 1000,
            },
        ],
    }


@freeze_time("2023-01-01 12:00:00")
class TestEnergyValleyEnv(unittest.TestCase):
    def setUp(self) -> None:
        with patch("marloes.data.util.read_series", return_value=pd.Series()):
            self.env = EnergyValley(config=get_new_config())

    def test_init(self):
        self.assertEqual(len(self.env.agents), 3)
        # check if the agents are of the right type
        self.assertIsInstance(self.env.agents[0], DemandAgent)
        self.assertIsInstance(self.env.agents[1], SolarAgent)
        self.assertIsInstance(self.env.agents[2], BatteryAgent)
        # check start time of each agent, should be equal to each other
        self.assertEqual(
            self.env.agents[0].asset.state.time, self.env.agents[1].asset.state.time
        )
        self.assertEqual(
            self.env.agents[0].asset.state.time, self.env.agents[2].asset.state.time
        )

    def test_agent_configurations(self):
        demand_agent = self.env.agents[0]
        solar_agent = self.env.agents[1]
        battery_agent = self.env.agents[2]

        self.assertEqual(demand_agent.asset.name, "Demand")
        self.assertEqual(demand_agent.asset.max_power_in, float("inf"))
        self.assertEqual(solar_agent.asset.name, "Solar")
        self.assertEqual(solar_agent.asset.max_power_out, float("inf"))
        self.assertEqual(battery_agent.asset.name, "Battery")
        self.assertEqual(battery_agent.asset.energy_capacity, 1000)
