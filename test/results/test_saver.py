import os
import unittest
from unittest.mock import patch, mock_open
import pandas as pd

from marloes.results.saver import Saver
from marloes.algorithms.base import AlgorithmType
from marloes.results.extractor import Extractor


class SaverTestCase(unittest.TestCase):
    @patch("os.makedirs")
    @patch("os.path.exists", return_value=False)
    @patch("builtins.open", new_callable=mock_open, read_data="0")
    def test_update_simulation_number(self, mock_open, mock_exists, mock_makedirs):
        algorithm = AlgorithmType.MODEL_BASED
        saver = Saver(algorithm)
        uid = saver._update_simulation_number()
        self.assertEqual(uid, 0)
        mock_open().write.assert_called_with("1")

    @patch("os.makedirs")
    @patch("os.path.exists", return_value=False)
    @patch("pandas.Series.to_csv")
    def test_save_metric(self, mock_to_csv, mock_exists, mock_makedirs):
        algorithm = AlgorithmType.MODEL_BASED
        saver = Saver(algorithm)
        series = pd.Series([1, 2, 3], index=["a", "b", "c"])
        saver._save_metric("test_metric", series)
        mock_makedirs.assert_called_once_with(
            os.path.join(saver.filename, "test_metric"), exist_ok=True
        )
        mock_to_csv.assert_called_once()

    @patch("os.makedirs")
    @patch("os.path.exists", return_value=False)
    @patch("pandas.Series.to_csv")
    def test_save(self, mock_to_csv, mock_exists, mock_makedirs):
        algorithm = AlgorithmType.MODEL_BASED
        saver = Saver(algorithm)
        extractor = Extractor()
        extractor.metrics = ["test_metric"]
        extractor.data = {"test_metric": pd.Series([1, 2, 3], index=["a", "b", "c"])}
        saver.save(extractor)
        mock_to_csv.assert_called_once()
