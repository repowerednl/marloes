import unittest
from freezegun import freeze_time
from datetime import datetime
from simon.assets.grid import Connection
from marloes.handlers.grid import GridHandler


@freeze_time("2023-01-01 12:00:00")
class TestGridHandler(unittest.TestCase):
    def setUp(self) -> None:
        self.grid_handler = GridHandler(config={}, start_time=datetime.now())

    def test_init(self):
        self.assertIsInstance(self.grid_handler.asset, Connection)
        self.assertEqual(self.grid_handler.asset.name, "Grid")
        self.assertEqual(self.grid_handler.asset.max_power_in, float("inf"))
        self.assertEqual(self.grid_handler.asset.max_power_out, float("inf"))
        # test the initial (default) state
        self.assertEqual(self.grid_handler.asset.state.time, datetime.now())
        self.assertEqual(self.grid_handler.asset.state.power, 0.0)

    def test_partial_init(self):
        partial_config = {
            "max_power_in": 100.0,
        }
        self.grid_handler = GridHandler(
            config=partial_config, start_time=datetime.now()
        )
        self.assertIsInstance(self.grid_handler.asset, Connection)
        self.assertEqual(self.grid_handler.asset.name, "Grid")
        self.assertEqual(self.grid_handler.asset.max_power_in, 100.0)
        self.assertEqual(self.grid_handler.asset.max_power_out, float("inf"))
        # test the initial (default) state
        self.assertEqual(self.grid_handler.asset.state.time, datetime.now())
        self.assertEqual(self.grid_handler.asset.state.power, 0.0)
