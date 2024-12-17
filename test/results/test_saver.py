import os
import unittest
import pandas as pd
import numpy as np

from marloes.results.saver import Saver
from marloes.results.extractor import Extractor

from unittest.mock import patch, mock_open


class SaverTestCase(unittest.TestCase):
    @patch("marloes.results.extractor.Extractor.__init__")
    def setUp(self, mock_extractor):
        self.config = {
            "algorithm": "model_based",
            "epochs": "100",
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
            ],
        }
        with patch("marloes.results.saver.Saver._save_config_to_yaml"):
            self.saver = Saver(self.config)
        self.saver.base_file_path = "test"
        self.saver.id_name = "test_uid.txt"
        mock_extractor.return_value = None
        self.extractor = Extractor()

    def tearDown(self):
        # reset the uid.txt file
        with open(os.path.join(self.saver.filename, self.saver.id_name), "w") as f:
            f.write("0")

    def test_update_simulation_number(self):
        uid = self.saver._update_simulation_number()
        self.assertEqual(uid, 0)
        uid = self.saver._update_simulation_number()
        self.assertEqual(uid, 1)

    @patch("pandas.Series.to_csv")
    @patch("builtins.open", new_callable=mock_open)
    def test_save_metric(self, mock_open, mock_to_csv):
        array = np.zeros(3)
        self.saver._save_metric("test_metric", array)
        mock_to_csv.assert_called()

    @patch("pandas.Series.to_csv")
    def test_save(self, mock_to_csv):
        self.saver.save(self.extractor)
        # how many times do we expect mock to be called, depends on Extractor
