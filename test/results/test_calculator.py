import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from marloes.results.calculator import Calculator


class TestCalculator(unittest.TestCase):
    @patch("marloes.results.calculator.Extractor.from_files")
    def setUp(self, mock_from_files):
        """
        Set up the Calculator instance and mock the from_files method.
        """
        mock_from_files.return_value = None
        self.calculator = Calculator(uid=1, dir="results")

    def test_get_metrics(self):
        """
        Test the get_metrics function.
        """
        metrics = ["loss", "total_battery_production"]
        self.calculator.extractor.loss = np.array([10, 20, 30])
        self.calculator.extractor.total_battery_production = np.array([5, 15, 25])

        results = self.calculator.get_metrics(metrics)
        expected_results = {
            "loss": np.array([10, 20, 30]),
            "total_battery_production": np.array([5, 15, 25]),
            "info": {"loss": [], "total_battery_production": []},
        }

        for key in metrics:
            np.testing.assert_array_equal(results[key], expected_results[key])
        self.assertEqual(results["info"], expected_results["info"])

    def test_sanity_check(self):
        """
        Test the _sanity_check function.
        """
        results = {
            "loss": np.array([10, 20, 30]),
            "total_battery_production": np.array([5, 15, 25]),
            "metric_not_found": None,
            "nan_metric": np.array([np.nan, 1, 2]),
            "too_long_metric": np.array([1] * (1 * 60 * 24 * 365 + 1) + [np.nan]),
        }

        info = self.calculator._sanity_check(results)
        expected_info = {
            "loss": [],
            "total_battery_production": [],
            "metric_not_found": ["metric_not_found is None."],
            "nan_metric": ["nan_metric contains NaN values."],
            "too_long_metric": [
                "too_long_metric is longer than a year.",
                "too_long_metric contains NaN values.",
            ],
        }

        self.assertEqual(info, expected_info)
