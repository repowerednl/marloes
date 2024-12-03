import unittest
from freezegun import freeze_time
from datetime import datetime
from simon.assets.grid import Connection
from marloes.agents.grid import GridAgent


@freeze_time("2023-01-01 12:00:00")
class TestGridAgent(unittest.TestCase):
    def setUp(self) -> None:
        self.grid_agent = GridAgent(config={}, start_time=datetime.now())

    def test_init(self):
        self.assertIsInstance(self.grid_agent.asset, Connection)
        self.assertEqual(self.grid_agent.asset.name, "Grid")
        self.assertEqual(self.grid_agent.asset.max_power_in, float("inf"))
        self.assertEqual(self.grid_agent.asset.max_power_out, float("inf"))
        # test the initial (default) state
        self.assertEqual(self.grid_agent.asset.state.time, datetime.now())
        self.assertEqual(self.grid_agent.asset.state.power, 0.0)

    def test_partial_init(self):
        partial_config = {
            "max_power_in": 100.0,
        }
        self.grid_agent = GridAgent(config=partial_config, start_time=datetime.now())
        self.assertIsInstance(self.grid_agent.asset, Connection)
        self.assertEqual(self.grid_agent.asset.name, "Grid")
        self.assertEqual(self.grid_agent.asset.max_power_in, 100.0)
        self.assertEqual(self.grid_agent.asset.max_power_out, float("inf"))
        # test the initial (default) state
        self.assertEqual(self.grid_agent.asset.state.time, datetime.now())
        self.assertEqual(self.grid_agent.asset.state.power, 0.0)

    def test_get_state(self):
        state = self.grid_agent.get_state()
        self.assertEqual(state.time, datetime.now())
        self.assertEqual(state.power, 0.0)
