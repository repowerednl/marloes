import unittest
from unittest.mock import patch
from freezegun import freeze_time

import pandas as pd

from marloes.simulation import Simulation
from marloes.algorithms.base import AlgorithmType


def get_new_config():  # function to return a new configuration, pop caused issues
    return {
        "algorithm": AlgorithmType.MODEL_BASED,
        "epochs": 100,
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
                "energy_capacity": 1000,
                "efficiency": 0.9,
                "max_power_in": 100,
                "max_power_out": 100,
            },
        ],
    }


class TestSimulation(unittest.TestCase):
    @patch("marloes.agents.solar.read_series", return_value=pd.Series())
    @patch("marloes.agents.demand.read_series", return_value=pd.Series())
    @patch("simon.assets.supply.Supply.load_default_state")
    @patch("simon.assets.demand.Demand.load_default_state")
    def setUp(self, *mocks) -> None:
        self.sim = Simulation(config=get_new_config())
        self.sim_saving = Simulation(config=get_new_config(), save_energy_flows=True)

    def test_init(self):
        # no saving
        self.assertEqual(self.sim.epochs, 100)
        self.assertEqual(len(self.sim.valley.agents), 3)
        self.assertFalse(self.sim.saving)
        self.assertIsNone(self.sim.flows, None)
        self.assertEqual(self.sim.algorithm, AlgorithmType.MODEL_BASED)
        # saving
        self.assertTrue(self.sim_saving.saving)
        self.assertIsInstance(self.sim_saving.flows, list)
