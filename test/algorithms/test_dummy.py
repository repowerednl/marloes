from unittest import TestCase
from unittest.mock import patch
import pytest

import pandas as pd
import numpy as np

from marloes.algorithms.dummy import Dummy
from marloes.agents.base import Agent


def get_new_config() -> dict:
    return {
        "algorithm": "dummy",
        "epochs": 10,
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
                "power": 100,
            },
        ],
    }


@pytest.mark.slow
class DummyTestCase(TestCase):
    """
    Dummy TestCase to check creation of an actual algorithm (and WorldModel in it).
    Not mocking anything to get the actual observations and actions.
    """

    @classmethod
    def setUpClass(cls) -> None:
        Agent._id_counters = {}
        with patch(
            "marloes.results.saver.Saver._update_simulation_number", return_value=0
        ):
            cls.alg = Dummy(config=get_new_config())

    def test_init(self):
        self.assertEqual(self.alg.epochs, 10)
        self.assertEqual(len(self.alg.environment.agents), 3)
        # environment should have observation_space torch.Size([2888])
        self.assertEqual(self.alg.environment.observation_space, (2888,))
        # environment should have action_space torch.Size([3])
        self.assertEqual(self.alg.environment.action_space, (3,))
