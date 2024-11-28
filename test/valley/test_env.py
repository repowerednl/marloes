import unittest
from freezegun import freeze_time
from datetime import datetime, timedelta
from marloes.agents.demand import DemandAgent
from marloes.agents.solar import SolarAgent
from marloes.valley.env import EnergyValley

CONFIG = {
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
    ],
}


@freeze_time("2023-01-01 12:00:00")
class TestEnergyValleyEnv(unittest.TestCase):
    def setUp(self) -> None:
        self.env = EnergyValley(config=CONFIG)

    def test_init(self):
        self.assertEqual(len(self.env.agents), 2)
        self.assertEqual(isinstance(self.env.agents[0], DemandAgent))
        self.assertEqual(isinstance(self.env.agents[1], SolarAgent))
