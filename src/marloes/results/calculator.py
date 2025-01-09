import numpy as np
import pandas as pd
from .extractor import Extractor
from marloes.valley.rewards.subrewards.base import SubReward
from marloes.valley.rewards.subrewards.co2 import CO2SubReward
from marloes.valley.rewards.subrewards.nb import NBSubReward
from marloes.valley.rewards.subrewards.nc import NCSubReward
from marloes.valley.rewards.subrewards.ss import SSSubReward
import logging


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
    }

    def __init__(self, uid: int | None = None, dir: str = "results"):
        self.extractor = Extractor(from_model=False)
        self.uid = self.extractor.from_files(uid, dir)

    def get_metrics(self, metrics: list[str]) -> dict[str, np.ndarray | None]:
        """
        Function to calculate the metrics.
        Returns a dictionary with the metrics as keys and the results as values,
        with an additional key 'info' for possible issues.
        """
        results = {
            metric: (
                self._get_reward_model(metric).calculate(self.extractor, actual=False)
                if self._get_reward_model(metric)
                else getattr(self.extractor, metric, None)
            )
            for metric in metrics
        }
        # Any other metrics are to be added here, so far no calculations needed because of the Reward Classes
        results["info"] = self._sanity_check(results)
        return results

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
