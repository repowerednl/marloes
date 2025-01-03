import numpy as np
from .extractor import Extractor
from marloes.valley.rewards.subrewards.base import SubReward
from marloes.valley.rewards.subrewards.co2 import CO2SubReward
from marloes.valley.rewards.subrewards.nb import NBSubReward
from marloes.valley.rewards.subrewards.nc import NCSubReward
from marloes.valley.rewards.subrewards.ss import SSSubReward


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
        self.extractor.from_files(uid, dir)

    def get_metrics(self, metrics: list[str]) -> list[np.ndarray | None]:
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

    def _sanity_check(
        self, results: dict[str, np.ndarray | None]
    ) -> dict[str, list[str]]:
        info = {}
        max_length = 1 * 60 * 24 * 365
        for key, value in results.items():
            issues = []
            if value is None:
                issues.append(f"{key} is None.")
            elif not isinstance(value, np.ndarray):
                issues.append(f"{key} is not a numpy array or None.")
            else:
                if len(value) > max_length:
                    issues.append(f"{key} is longer than a year.")
                if np.nan in value:
                    issues.append(f"{key} contains NaN values.")
            info[key] = issues
        return info
