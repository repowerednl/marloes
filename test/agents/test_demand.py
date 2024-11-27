import unittest
from freezegun import freeze_time
from datetime import datetime, timedelta
from simon.assets.demand import Demand
from simon.datasource.data_source import (
    DummyDataSource,
)
from marloes.agents.demand import DemandAgent

CONFIG = {
    "name": "DemandOne",
    "profile": "Farm",
    "scale": 1,
    "max_power_in": 10.0,
    "constant_demand": 5.0,
    "curtailable_by_solver": True,
    "upward_dispatchable": False,
}


@freeze_time("2023-01-01 12:00:00")
class TestDemandAgent(unittest.TestCase):
    def setUp(self) -> None:
        self.demand_agent = DemandAgent(start_time=datetime.now(), config=CONFIG)

    def test_init(self):
        self.assertIsInstance(self.demand_agent.asset, Demand)
        self.assertEqual(self.demand_agent.asset.name, "DemandOne")
        self.assertEqual(self.demand_agent.asset.max_power_in, 10.0)
        self.assertIsInstance(self.demand_agent.asset.data_source, DummyDataSource)
        self.assertEqual(self.demand_agent.asset.data_source.value, 5.0)

    def test_partial_init(self):
        partial_config = {}
        self.demand_agent = DemandAgent(
            start_time=datetime.now(), config=partial_config
        )
        self.assertIsInstance(self.demand_agent.asset, Demand)
        self.assertEqual(self.demand_agent.asset.name, "Demand")
        self.assertFalse(self.demand_agent.asset.curtailable_by_solver)
