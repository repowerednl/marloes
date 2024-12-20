import os
import unittest
import pandas as pd
import numpy as np

from marloes.results.saver import Saver
from marloes.results.extractor import Extractor

from unittest.mock import patch, mock_open, MagicMock


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
        with patch("marloes.results.saver.Saver._save_config_to_yaml"), patch(
            "marloes.results.saver.Saver._update_simulation_number", return_value=0
        ), patch("marloes.results.saver.Saver._validate_folder"):
            self.saver = Saver(self.config, test=True)
        self.saver.base_file_path = "test"
        self.saver.id_name = "test_uid.txt"
        self.extractor = Extractor()
        self.extractor.test_metric = np.array([0.1, 0.2, 0.3])
        self.extractor.i = 2  # current iteration index

    def tearDown(self):
        # after testing, also reset the test_uid.txt file to 0
        with open("test/test_uid.txt", "w") as f:
            f.write("0")

    @patch("numpy.save")
    @patch("numpy.load")
    @patch("os.path.exists")
    def test_save_metric(self, mock_exists, mock_load, mock_save):
        mock_exists.return_value = False
        array = np.zeros(3)
        self.saver._save_metric("test_metric", array)
        # Manually extract the arguments passed to numpy.save
        args, _ = mock_save.call_args

        # Expected file path and data
        expected_path = os.path.join(
            self.saver.base_file_path,
            "test_metric",
            f"{self.saver.uid}.npy",
        )
        expected_data = array

        # Assert that the file path matches
        assert args[0] == expected_path

        # Assert that the numpy array matches
        np.testing.assert_array_equal(args[1], expected_data)

    @patch("numpy.save")
    def test_save(self, mock_save):
        self.saver.save(self.extractor)

        # Manually extract the arguments passed to numpy.save
        args, _ = mock_save.call_args

        # Expected file path and data
        expected_path = os.path.join(
            self.saver.base_file_path,
            "test_metric",
            f"{self.saver.uid}.npy",
        )
        expected_data = np.array([0.1, 0.2])

        # Assert that the file path matches
        assert args[0] == expected_path

        # Assert that the numpy array matches
        np.testing.assert_array_equal(args[1], expected_data)

    def test_update_simulation_number(self):
        uid = self.saver._update_simulation_number()
        self.assertEqual(uid, 0)
        uid = self.saver._update_simulation_number()
        self.assertEqual(uid, 1)
