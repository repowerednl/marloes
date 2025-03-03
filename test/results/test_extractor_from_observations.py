import unittest
from unittest.mock import patch
from collections import defaultdict
import pandas as pd

from marloes.agents.base import Agent
from marloes.algorithms.priorities import Priorities
from marloes.results.extractor import Extractor

from test.util import get_new_config

MINUTES_IN_A_YEAR = 525600


class TestExtractorFromObservations(unittest.TestCase):
    @classmethod
    @patch("marloes.agents.solar.read_series", return_value=pd.Series())
    @patch("marloes.agents.demand.read_series", return_value=pd.Series())
    @patch("marloes.agents.wind.read_series", return_value=pd.Series())
    @patch("simon.assets.supply.Supply.load_default_state")
    @patch("simon.assets.demand.Demand.load_default_state")
    @patch("simon.assets.wind.Wind.load_default_state")
    def setUp(cls, *mocks) -> None:
        config = get_new_config()
        agents_list = config["agents"]
        agents_list.append(
            {
                "type": "wind",
                "location": "Onshore",
                "AC": 900,
                "power": 1000,
            }
        )
        config["agents"] = agents_list
        config["algorithm"] = "priorities"

        with patch("marloes.results.saver.Saver._save_config_to_yaml"), patch(
            "marloes.results.saver.Saver._update_simulation_number", return_value=0
        ), patch("marloes.results.saver.Saver._validate_folder"):
            Agent._id_counters = {}
            cls.alg = Priorities(config=config)
        # make sure there are 5 + 1 agents in the config
        assert len(cls.alg.environment.agents) == 6

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
