import unittest
from collections import defaultdict

from marloes.factories import EnergyValleyFactory

from simon.solver import Model

from marloes.results.extractor import Extractor

MINUTES_IN_A_YEAR = 525600


class TestExtractorFromObservations(unittest.TestCase):
    @classmethod
    def setUp(cls):
        """
        To test _from_observations we need valid observations (from an initialized algorithm)
        """
        cls.valley = EnergyValleyFactory()

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
