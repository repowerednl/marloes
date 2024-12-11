import os
import unittest
from unittest.mock import patch, mock_open
import pandas as pd
import yaml

from marloes.results.saver import Saver
from marloes.algorithms.base import AlgorithmType
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
        self.series = pd.Series([1, 2, 3], index=["a", "b", "c"])
        self.extractor = Extractor()
        self.extractor.metrics = ["test_metric"]
        self.extractor.data = {"test_metric": self.series}

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

    @patch("os.makedirs")
    @patch("os.path.exists", return_value=False)
    @patch("builtins.open", new_callable=mock_open)
    def test_save_config_to_yaml(self, mock_open, mock_exists, mock_makedirs):
        self.saver._save_config_to_yaml(self.config)
        mock_open.assert_called_once_with(
            os.path.join(
                self.saver.filename, "configs", f"0_{self.config['algorithm']}.yaml"
            ),
            "w",
        )
        mock_open().write.assert_called()
        self.assertEqual(
            mock_open().write.call_count, 58
        )  # 58 is the length of the yaml.dump(self.config, f), writing all special characters as well

    @patch("os.makedirs")
    @patch("os.path.exists", return_value=False)
    @patch("pandas.Series.to_csv")
    def test_save_metric(self, mock_to_csv, mock_exists, mock_makedirs):
        self.saver._save_metric("test_metric", self.series)
        mock_makedirs.assert_called_once_with(
            os.path.join(self.saver.filename, "test_metric"), exist_ok=True
        )
        mock_to_csv.assert_called_once_with(
            os.path.join(self.saver.filename, "test_metric", "0_model_based.csv"),
            mode="a",
            header=True,
            index=False,
        )

    @patch("os.makedirs")
    @patch("os.path.exists", return_value=False)
    @patch("pandas.Series.to_csv")
    def test_save(self, mock_to_csv, mock_exists, mock_makedirs):
        self.saver.save(self.extractor)
        mock_to_csv.assert_called_once_with(
            os.path.join(self.saver.filename, "test_metric", "0_model_based.csv"),
            mode="a",
            header=True,
            index=False,
        )
