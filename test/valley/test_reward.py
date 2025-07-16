from datetime import datetime
import unittest
from unittest.mock import patch
import numpy as np
from marloes.factories import ExtractorFactory, RewardFactory
from marloes.valley.rewards.reward import Reward


def get_new_config() -> dict:
    return {
        "algorithm": "PrioFlow",
        "training_steps": 100,
        "handlers": [
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
        "replay_buffers": {
            "real_capacity": 1000,
            "model_capacity": 1000,
        },
    }


config = get_new_config()


class TestReward(unittest.TestCase):
    def setUp(self):
        """
        Set up a mock Extractor and default Reward instance.
        """
        self.extractor = ExtractorFactory()

        # Default scaling factors
        self.default_scaling = {"active": True, "scaling_factor": 1.0}

        self.timestamp = datetime(2023, 10, 1, 12, 0, 0)

    def test_ss_reward(self):
        """
        Test the self-sufficiency penalty.
        """
        ## Test actual
        reward = Reward(config, actual=True, SS=self.default_scaling)

        # For SS we have to simulate the loop
        result = reward.get(self.extractor, self.timestamp)

        # Cumulative grid state: -10 + 5 + 10 -> max(0, cumulative)
        # Cumulative is skipped because no in between updates: 10 * -1 = -10
        expected = -10
        self.assertAlmostEqual(result, expected, places=5)

        ## Test not actual
        reward = Reward(actual=False, SS=self.default_scaling)
        result = reward.get(self.extractor, self.timestamp)

        # Cumulative grid state: [-10, 5, 10] -> max(0, cumulative)
        expected = -np.maximum(0, np.cumsum(self.extractor.grid_state))
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_nc_reward(self):
        """
        Test the net-congestion penalty.
        """
        ## Test actual
        reward = Reward(config, actual=True, NC=self.default_scaling)
        result = reward.get(self.extractor, self.timestamp)

        # Latest grid state (negative part only): min(0, 10) -> 0
        expected = np.minimum(0, self.extractor.grid_state[-1])
        self.assertAlmostEqual(result, expected, places=5)

        ## Test not actual
        reward = Reward(actual=False, NC=self.default_scaling)
        result = reward.get(self.extractor, self.timestamp)

        # Grid state: [-10, 5, 10] -> min(0, grid_state)
        expected = np.minimum(0, self.extractor.grid_state)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_ne_reward(self):
        """
        Test the Nomination Error reward. Only calculates every hour so we need to add values to the extractor.
        """
        # Add values to the extractor using post_generation method
        new_extractor = ExtractorFactory()
        # change values:
        new_extractor.total_solar_production = np.array([30] * 61)
        new_extractor.total_solar_nomination = np.array([20] * 61)
        new_extractor.total_wind_production = np.array([25] * 61)
        new_extractor.total_wind_nomination = np.array([30] * 61)
        new_extractor.total_demand = np.array([-39] * 61)
        new_extractor.total_demand_nomination = np.array([-40] * 61)
        new_extractor.total_nomination_fraction = np.array([0.5] * 61)
        new_extractor.grid_state = np.array([10] * 61)
        new_extractor.i = 60

        ## Test actual (full hour)
        reward = Reward(config, actual=True, NE=self.default_scaling)
        result = reward.get(new_extractor)
        # total production = 30 + 25 - 39 = 16 * 60 = 960
        # total nomination = 20 + 30 - 40 = 10 * 60 = 600
        expected = -(abs(960 - 600) * 1)
        self.assertEqual(result, expected)

        # Test actual (intermediate)
        new_extractor.i = 30
        reward = Reward(config, actual=True, NE=self.default_scaling)
        result = reward.get(new_extractor)
        # total nomination = 30 + 20 - 40 = 10
        # expected nomination fraction = 10 / 60 * (30 % 60) = 5
        expected_nomination_fraction = 10 / 60 * (new_extractor.i % 60)
        # intermediate scaling factor is default_scaling / 60 = 1/60
        expected_penalty = -abs(expected_nomination_fraction - 0.5) * 1 / 60
        self.assertAlmostEqual(result, expected_penalty, places=5)

        ## Test not actual
        reward = Reward(config, actual=False, NE=self.default_scaling)
        result = reward.get(new_extractor)
        # result should be np.array, with all zeros except for the hour where the penalty is calculated
        expected = np.zeros(61)
        expected[60] = -(abs(960 - 600) * 1)
        self.assertEqual(len(result), len(expected))
        self.assertEqual(result[60], expected[60])

    def test_total_reward(self):
        """
        Test the total reward calculation with multiple active sub-rewards.
        """
        ## Test actual
        reward = Reward(
            config,
            actual=True,
            SS={"active": True, "scaling_factor": 1.5},
            NC={"active": True, "scaling_factor": 1.0},
            NB={"active": True, "scaling_factor": 0.5},
        )
        result = reward.get(self.extractor, self.timestamp)

        # Sub-rewards:
        ss_penalty = 1.5 * -10
        nc_penalty = 1.0 * np.minimum(0, self.extractor.grid_state[-1])
        nb_reward = 0.5 * -10

        # Total reward
        expected = ss_penalty + nc_penalty + nb_reward

        self.assertAlmostEqual(result, expected, places=5)

        ## Test not actual
        reward = Reward(
            config,
            actual=False,
            SS={"active": True, "scaling_factor": 1.5},
            NC={"active": True, "scaling_factor": 1.0},
            NB={"active": True, "scaling_factor": 0.5},
        )
        result = reward.get(self.extractor, self.timestamp)

        # Sub-rewards:

        ss_penalty = 1.5 * -np.maximum(0, np.cumsum(self.extractor.grid_state))
        nc_penalty = 1.0 * np.minimum(0, self.extractor.grid_state)
        nb_reward = 0.5 * -np.cumsum(self.extractor.grid_state)

        # Total reward
        expected = ss_penalty + nc_penalty + nb_reward

        np.testing.assert_array_almost_equal(result, expected, decimal=5)
