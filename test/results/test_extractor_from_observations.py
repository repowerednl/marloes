import unittest
from unittest.mock import patch
from collections import defaultdict
import pandas as pd

from marloes.results.extractor import Extractor

from test.util import get_mock_observation

MINUTES_IN_A_YEAR = 525600


class TestExtractorFromObservations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # initialize the Extractor
        cls.extractor = Extractor(from_model=True, chunk_size=1000)
        cls.base_observation = get_mock_observation(
            solar_power=2.0,  # 3*2.0 = 6.0
            wind_power=1.0,  # 3*1.0 = 3.0
            solar_nomination=1.8,  # 1.8
            wind_nomination=0.9,  # 1.8
        )

    def test__get_total_nomination_per_type(self):
        """
        Test that _get_total_nomination_per_type correctly returns the sum of nominations per type.
        Observations can be passed as a dict with agent_id as key and observation as value.
        """
        self.extractor.clear()
        # Summing the observations per type should return for Solar: 1.8 and for Wind: 0.9
        result = self.extractor._get_total_nomination_by_type(self.base_observation)
        self.assertEqual(result["Solar"], 1.8)
        self.assertEqual(result["Wind"], 0.9)

        # Add a solar and a wind agent with different nominations
        new_observation = self.base_observation.copy()
        new_observation.update(
            {
                "SolarAgent 1": {
                    "forecast": [2.0] * 3,
                    "power": 2.0,
                    "available_power": 0.1,
                    "nomination": 2.2,
                },
                "WindAgent 1": {
                    "forecast": [1.0] * 3,
                    "power": 1.0,
                    "available_power": 0.1,
                    "nomination": 1.5,
                },
            }
        )

        result = self.extractor._get_total_nomination_by_type(new_observation)
        self.assertEqual(result["Solar"], 4.0)
        self.assertEqual(result["Wind"], 2.4)

    def test_from_observations(self):
        """
        Test that Extractor.from_observations correctly aggregates per type (and saves it as attribute)
        - nominations
        """
        self.extractor.clear()
        self.extractor.from_observations(self.base_observation)
        self.assertEqual(self.extractor.total_solar_nomination[0], 1.8)
        self.assertEqual(self.extractor.total_wind_nomination[0], 0.9)

        # Add a solar and a wind agent with different nominations
        new_observation = self.base_observation.copy()
        new_observation.update(
            {
                "SolarAgent 1": {
                    "forecast": [2.0] * 3,
                    "power": 2.0,
                    "available_power": 0.1,
                    "nomination": 2.2,
                },
                "WindAgent 1": {
                    "forecast": [1.0] * 3,
                    "power": 1.0,
                    "available_power": 0.1,
                    "nomination": 1.5,
                },
            }
        )
        # Somehow two assets where added after the first timestep, we update and also check the previous value
        self.extractor.update()

        self.extractor.from_observations(new_observation)
        self.assertEqual(self.extractor.total_solar_nomination[0], 1.8)
        self.assertEqual(self.extractor.total_wind_nomination[0], 0.9)
        self.assertEqual(self.extractor.total_solar_nomination[1], 4.0)
        self.assertEqual(self.extractor.total_wind_nomination[1], 2.4)
