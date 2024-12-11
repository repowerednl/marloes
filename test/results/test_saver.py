import os
import unittest
from unittest.mock import patch, mock_open
import pandas as pd
import numpy as np

from marloes.results.saver import Saver
from marloes.results.extractor import Extractor


class SaverTestCase(unittest.TestCase):
    def setUp(self):
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
        self.saver = Saver(self.config)
        self.extractor = Extractor()

    def tearDown(self):
        # reset the uid.txt file
        with open(os.path.join(self.saver.filename, "uid.txt"), "w") as f:
            f.write("0")

    @patch("os.makedirs")
    @patch("os.path.exists", return_value=False)
    @patch("builtins.open", new_callable=mock_open, read_data="0")
    def test_update_simulation_number(self, mock_open, mock_exists, mock_makedirs):
        uid = self.saver._update_simulation_number()
        self.assertEqual(uid, 0)
        mock_open().write.assert_called_with("1")

    @patch("builtins.open", new_callable=mock_open)
    def test_save_metric(self, mock_open):
        self.saver._validate_folder("metric")
        array = np.zeros(3)
        self.saver._save_metric("metric", array)
        mock_open().write.assert_called()

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.makedirs")
    def test_save(self, mock_makedirs, mock_open):
        self.saver.save(self.extractor)
        mock_open().write.assert_called()

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.makedirs")
    def test_save_config_to_yaml(self, mock_makedirs, mock_open):
        self.saver._save_config_to_yaml(self.config)
        mock_open().write.assert_called()
