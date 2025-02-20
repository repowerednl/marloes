import unittest
from unittest.mock import patch
from collections import defaultdict

from simon.solver import Model

from marloes.valley.env import EnergyValley
from marloes.results.extractor import Extractor

from test.util import get_new_config

MINUTES_IN_A_YEAR = 525600


@patch("marloes.agents.solar.read_series")
@patch("marloes.agents.wind.read_series")
@patch("marloes.agents.demand.read_series")
class TestExtractorFromObservations(unittest.TestCase):
    @classmethod
    def setUp(cls):
        """
        To test _from_observations we need valid observations (from an initialized algorithm)
        """
        # creating a valley with Priorities algorithm
        config = get_new_config()
        # add a wind agent to config
        config_with_wind = config["agents"].append(
            {
                "type": "wind",
                "location": "Onshore",
                "power": 1400,
                "AC": 1200,
            }
        )

        cls.valley = EnergyValley(config_with_wind, "Priorities")
        # make sure there are 5 + 1 agents in the config
        assert len(cls.valley.agents) == 6

    def test_from_observations(self):
        """
        Test that Extractor.from_observations correctly aggregates
        - nominations
        """
        pass

    def test__get_total_nomination_per_type(self):
        """
        Test that _get_total_nomination_per_type correctly returns the sum of nominations per type.
        """
        pass
