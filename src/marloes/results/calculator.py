import logging

import numpy as np
import pandas as pd

from marloes.valley.rewards.subrewards import (
    CO2SubReward,
    NBSubReward,
    NCSubReward,
    SSSubReward,
    SubReward,
)
from marloes.valley.rewards.subrewards.ne import NESubReward

from .extractor import Extractor


class Calculator:
    """
    Calculates/Gathers the metrics passed in a list.
    It initializes the Extractor upon creation and uses it to 'calculate' the metrics.
    """

    REWARD_CLASSES = {
        "CO2": CO2SubReward,
        "SS": SSSubReward,
        "NC": NCSubReward,
        "NB": NBSubReward,
        "NE": NESubReward,
    }
    EXTRA_METRICS = {
        "grid_state": "cumulative_grid_state",
    }

    def __init__(self, uid: int | None = None, dir: str = "results"):
        self.extractor = Extractor(from_model=False)
        self.uid = self.extractor.from_files(uid, dir)

    def get_all_metrics(self) -> list[str]:
        """
        Returns a list of all available metrics.
        This is useful to check which metrics are available for plotting or analysis.
        """
        base_metrics = self.extractor.get_all_metrics()
        # Extract al extra metrics for which the key is in the base metrics
        extra_metrics = [
            self.EXTRA_METRICS[key] for key in self.EXTRA_METRICS if key in base_metrics
        ]
        return base_metrics + extra_metrics

    def get_metrics(self, metrics: list[str]) -> dict[str, np.ndarray | None]:
        """
        Function to calculate the metrics.
        Returns a dictionary with the metrics as keys and the results as values,
        with an additional key 'info' for possible issues.
        """
        results = {}

        for metric in metrics:
            # Option 1: Reward
            reward_model = self._get_reward_model(metric)
            if reward_model:
                results[metric] = reward_model.calculate(self.extractor, actual=False)
                continue

            # Option 2: Custom method requires calculation in the Calculator
            if hasattr(self, metric) and callable(getattr(self, metric)):
                method = getattr(self, metric)
                results[metric] = method()
                continue

            # Option 3: Just extractor metric
            results[metric] = getattr(self.extractor, metric, None)

        results["info"] = self._sanity_check(results)
        return results

    def cumulative_grid_state(self) -> np.ndarray:
        """
        Calculates the cumulative grid state.
        """
        return np.cumsum(self.extractor.grid_state)

    def _get_reward_model(self, metric: str) -> SubReward | None:
        reward_class = self.REWARD_CLASSES.get(metric)
        return reward_class(active=True, scaling_factor=1) if reward_class else None

    @staticmethod
    def _sanity_check(results: dict[str, np.ndarray | None]) -> dict[str, list[str]]:
        info = {}
        max_length = 1 * 60 * 24 * 365
        for key, value in results.items():
            issues = []
            if value is None:
                issues.append(f"{key} is None.")
            elif not (isinstance(value, np.ndarray) or isinstance(value, pd.DataFrame)):
                issues.append(f"{key} is not a numpy array or pandas DataFrame.")
            else:
                if len(value) > max_length:
                    issues.append(f"{key} is longer than a year.")
                if isinstance(value, np.ndarray) and np.isnan(value).any():
                    issues.append(f"{key} contains NaN values.")
                if isinstance(value, pd.DataFrame) and value.isnull().values.any():
                    issues.append(f"{key} contains NaN values.")
            info[key] = issues
        return info

    @staticmethod
    def log_sanity_check(info_dict: dict[str, list[str]]):
        """
        Logs the info dictionary from the Calculator's sanity check.
        """
        for metric, issues in info_dict.items():
            if not issues:
                logging.info(f"No issues found for metric '{metric}'.")
            else:
                # Log each issue for this metric
                for issue in issues:
                    logging.info(f"Issue for metric '{metric}': {issue}")
