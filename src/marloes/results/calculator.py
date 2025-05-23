from datetime import datetime
import logging
import os

import numpy as np
import pandas as pd
import yaml

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
        "grid_state": ["cumulative_grid_state"],
        "reward": ["cumulative_reward", "reward_daily"],
        "total_grid_production": [
            "cumulative_grid_production",
            "daily_grid_production",
        ],
        "total_solar_production": ["surplus"],
    }

    def __init__(self, uid: int | None = None, dir: str = "results"):
        self.extractor = Extractor(from_model=False)
        self.uid = self.extractor.from_files(uid, dir)

        # Get start time from dir/configs/uid.yaml, loading the yaml to dict and extracting the start time
        if os.path.exists(f"{dir}/configs/{uid}.yaml"):
            with open(f"{dir}/configs/{uid}.yaml", "r") as f:
                config: dict = yaml.safe_load(f)
            start_time = config.get("start_time", datetime(2025, 1, 1, tzinfo=None))
            self.start_time = pd.to_datetime(start_time, utc=True)

    def get_all_metrics(self) -> list[str]:
        """
        Returns a list of all available metrics.
        This is useful to check which metrics are available for plotting or analysis.
        """
        base_metrics = self.extractor.get_all_metrics()
        # Extract al extra metrics for which the key is in the base metrics
        extra_metrics = []
        for key in self.EXTRA_METRICS:
            if key in base_metrics:
                extra_metrics.extend(self.EXTRA_METRICS[key])
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
        results["start_time"] = self.start_time
        return results

    def cumulative_grid_state(self) -> np.ndarray:
        """
        Calculates the cumulative grid state.
        """
        return np.cumsum(self.extractor.grid_state)

    def cumulative_reward(self) -> np.ndarray:
        """
        Calculates the cumulative reward.
        """
        return np.cumsum(self.extractor.reward)

    def cumulative_grid_production(self) -> np.ndarray:
        """
        Calculates the cumulative grid production.
        """
        return np.cumsum(self.extractor.total_grid_production)

    def reward_daily(self) -> np.ndarray:
        """
        Shows the daily improvement of the reward.
        """
        series = pd.Series(self.extractor.reward)
        index = pd.date_range(
            start="2025-01-01",
            periods=len(series),
            freq="min",
        )
        series.index = index
        series = series.resample("D").sum()
        return series.values

    def daily_grid_production(self) -> np.ndarray:
        """
        Shows the daily improvement of the grid production.
        """
        series = pd.Series(self.extractor.total_grid_production)
        index = pd.date_range(
            start="2025-01-01",
            periods=len(series),
            freq="min",
        )
        series.index = index
        series = series.resample("D").sum()
        return series.values

    def surplus(self) -> np.ndarray:
        """
        Calculates the surplus.
        """
        return self.extractor.total_solar_production - self.extractor.total_demand

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
                    num_nan = np.isnan(value).sum()
                    issues.append(f"{key} contains {num_nan} NaN values.")
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
