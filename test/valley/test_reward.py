import unittest
from unittest.mock import patch
import numpy as np
from marloes.factories import ExtractorFactory, RewardFactory
from marloes.valley.rewards.reward import Reward


class TestReward(unittest.TestCase):
    def setUp(self):
        """
        Set up a mock Extractor and default Reward instance.
        """
        self.extractor = ExtractorFactory()

        # Default scaling factors
        self.default_scaling = {"active": True, "scaling_factor": 1.0}

    def test_co2_reward(self):
        """
        Test the CO2 penalty calculation.
        """
        ## Test actual
        # Also test whether not passing a scaling factor defaults to 1.0
        reward = Reward(actual=True, CO2={"active": True})
        reward.sub_rewards[
            "CO2"
        ].EMISSION_COEFFICIENTS = RewardFactory.EMISSION_COEFFICIENTS
        result = reward.get(self.extractor)

        # Expected CO2 penalty:
        # - (solar: 30*0.2 + wind: 25*0.1 + battery: 20*0.3 + grid: 15*0.5)
        expected = -(30 * 0.2 + 25 * 0.1 + 20 * 0.3 + 15 * 0.5)
        self.assertAlmostEqual(result, expected, places=5)

        ## Test not actual
        reward = Reward(actual=False, CO2=self.default_scaling)
        reward.sub_rewards[
            "CO2"
        ].EMISSION_COEFFICIENTS = RewardFactory.EMISSION_COEFFICIENTS
        result = reward.get(self.extractor)

        # Expected CO2 penalty:
        # - (solar: [10, 20, 30]*0.2 + wind: [5, 15, 25]*0.1 + battery: [0, 10, 20]*0.3 + grid: [-5, 5, 15]*0.5)
        # Expect to return array of penalties summed per index
        expected = -np.array(
            [
                10 * 0.2 + 5 * 0.1 + 0 * 0.3 + -5 * 0.5,
                20 * 0.2 + 15 * 0.1 + 10 * 0.3 + 5 * 0.5,
                30 * 0.2 + 25 * 0.1 + 20 * 0.3 + 15 * 0.5,
            ]
        )
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_ss_reward(self):
        """
        Test the self-sufficiency penalty.
        """
        ## Test actual
        reward = Reward(actual=True, SS=self.default_scaling)

        # For SS we have to simulate the loop
        result = reward.get(self.extractor)

        # Cumulative grid state: -10 + 5 + 10 -> max(0, cumulative)
        # Cumulative is skipped because no in between updates: 10 * -1 = -10
        expected = -10
        self.assertAlmostEqual(result, expected, places=5)

        ## Test not actual
        reward = Reward(actual=False, SS=self.default_scaling)
        result = reward.get(self.extractor)

        # Cumulative grid state: [-10, 5, 10] -> max(0, cumulative)
        expected = -np.maximum(0, np.cumsum(self.extractor.grid_state))
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_nc_reward(self):
        """
        Test the net-congestion penalty.
        """
        ## Test actual
        reward = Reward(actual=True, NC=self.default_scaling)
        result = reward.get(self.extractor)

        # Latest grid state (negative part only): min(0, 10) -> 0
        expected = np.minimum(0, self.extractor.grid_state[-1])
        self.assertAlmostEqual(result, expected, places=5)

        ## Test not actual
        reward = Reward(actual=False, NC=self.default_scaling)
        result = reward.get(self.extractor)

        # Grid state: [-10, 5, 10] -> min(0, grid_state)
        expected = np.minimum(0, self.extractor.grid_state)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_nb_reward(self):
        """
        Test the net-balance reward.
        """
        ## Test actual
        reward = Reward(actual=True, NB=self.default_scaling)
        result = reward.get(self.extractor)

        # Cumulative grid state: -10 + 5 + 10
        # Again, cumulative is skipped because no in between updates: -10
        expected = -(10)
        self.assertAlmostEqual(result, expected, places=5)

        ## Test not actual
        reward = Reward(actual=False, NB=self.default_scaling)
        result = reward.get(self.extractor)

        # Cumulative grid state: [-10, 5, 10]
        expected = -np.cumsum(self.extractor.grid_state)
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
        new_extractor.total_demand_nomination = np.array([-40] * 61)
        new_extractor.total_nomination_fraction = np.array([0.5] * 61)
        new_extractor.grid_state = np.array([10] * 61)
        new_extractor.i = 60

        ## Test actual (full hour)
        reward = Reward(actual=True, NE=self.default_scaling)
        result = reward.get(new_extractor)
        # total solar production: mean(30 * 60) = 30, total solar nomination: for that hour at every timestep 20
        # total wind production: mean(25 * 60) = 25, total wind nomination: for that hour at every timestep 30
        expected_solar_penalty = abs(30 - 20)
        expected_wind_penalty = abs(25 - 30)
        expected = -(expected_solar_penalty + expected_wind_penalty)
        self.assertEqual(result, expected)

        # Test actual (intermediate)
        new_extractor.i = 30
        reward = Reward(actual=True, NE=self.default_scaling)
        result = reward.get(new_extractor)
        # total nomination = 30 + 20 - 40 = 10
        # expected nomination fraction = 10 / 60 * (30 % 60) = 5
        expected_nomination_fraction = 10 / 60 * (new_extractor.i % 60)
        # intermediate scaling factor is default_scaling / 60 = 1/60
        expected_penalty = -abs(expected_nomination_fraction - 0.5) * 1 / 60
        self.assertAlmostEqual(result, expected_penalty, places=5)

        ## Test not actual
        reward = Reward(actual=False, NE=self.default_scaling)
        result = reward.get(new_extractor)
        # result should be np.array, with all zeros except for the hour where the penalty is calculated
        expected = np.zeros(61)
        expected[60] = -(expected_solar_penalty + expected_wind_penalty)
        self.assertEqual(len(result), len(expected))
        self.assertEqual(result[60], expected[60])

    def test_total_reward(self):
        """
        Test the total reward calculation with multiple active sub-rewards.
        """
        ## Test actual
        reward = Reward(
            actual=True,
            CO2={"active": True, "scaling_factor": 2.0},
            SS={"active": True, "scaling_factor": 1.5},
            NC={"active": True, "scaling_factor": 1.0},
            NB={"active": True, "scaling_factor": 0.5},
        )
        reward.sub_rewards[
            "CO2"
        ].EMISSION_COEFFICIENTS = RewardFactory.EMISSION_COEFFICIENTS
        result = reward.get(self.extractor)

        # Sub-rewards:
        co2_penalty = 2.0 * -(30 * 0.2 + 25 * 0.1 + 20 * 0.3 + 15 * 0.5)
        ss_penalty = 1.5 * -10
        nc_penalty = 1.0 * np.minimum(0, self.extractor.grid_state[-1])
        nb_reward = 0.5 * -10

        # Total reward
        expected = co2_penalty + ss_penalty + nc_penalty + nb_reward

        self.assertAlmostEqual(result, expected, places=5)

        ## Test not actual
        reward = Reward(
            actual=False,
            CO2={"active": True, "scaling_factor": 2.0},
            SS={"active": True, "scaling_factor": 1.5},
            NC={"active": True, "scaling_factor": 1.0},
            NB={"active": True, "scaling_factor": 0.5},
        )
        reward.sub_rewards[
            "CO2"
        ].EMISSION_COEFFICIENTS = RewardFactory.EMISSION_COEFFICIENTS
        result = reward.get(self.extractor)

        # Sub-rewards:
        co2_penalty = (
            -np.array(
                [
                    10 * 0.2 + 5 * 0.1 + 0 * 0.3 + -5 * 0.5,
                    20 * 0.2 + 15 * 0.1 + 10 * 0.3 + 5 * 0.5,
                    30 * 0.2 + 25 * 0.1 + 20 * 0.3 + 15 * 0.5,
                ]
            )
            * 2.0
        )
        ss_penalty = 1.5 * -np.maximum(0, np.cumsum(self.extractor.grid_state))
        nc_penalty = 1.0 * np.minimum(0, self.extractor.grid_state)
        nb_reward = 0.5 * -np.cumsum(self.extractor.grid_state)

        # Total reward
        expected = co2_penalty + ss_penalty + nc_penalty + nb_reward

        np.testing.assert_array_almost_equal(result, expected, decimal=5)
