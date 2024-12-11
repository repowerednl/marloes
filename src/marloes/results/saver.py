import os
import yaml
import pandas as pd

from .extractor import Extractor


class Saver:
    """
    Class that saves the data extracted by the Extractor
    """

    def __init__(self, config: dict) -> None:
        self.name = "Saver"
        self.algorithm = config["algorithm"]
        self.filename = "results"
        self.uid = self._update_simulation_number()
        self._save_config_to_yaml(config)

    def save(self, extractor: Extractor) -> None:
        for metric in extractor.metrics:  # or extractor.get_metrics()
            # series = extractor.data[metric]
            series = pd.Series([1, 2, 3], index=["a", "b", "c"])
            self._save_metric(metric, series)

    def _save_config_to_yaml(self, config: dict) -> None:
        """
        Function that saves the configuration to a yaml file
        """
        config_files = os.path.join(self.filename, "configs")
        os.makedirs(config_files, exist_ok=True)
        config_filename = os.path.join(
            config_files, f"{self.uid}_{self.algorithm}.yaml"
        )
        with open(config_filename, "w") as f:
            yaml.dump(config, f)

    def save_model(self, model) -> None:
        pass

    def _update_simulation_number(self) -> int:
        """
        Function that extracts and updates the uid.txt file in root folder/results/uid.txt
        """
        uid_path = os.path.join(self.filename, "uid.txt")
        with open(uid_path, "r+") as f:
            uid = int(f.read())
            f.seek(0)
            f.write(str(uid + 1))
            f.truncate()
        return uid

    def _validate_folder(self, metric: str) -> None:
        """
        Function that validates the folder for a single metric, if it does not exist, it is created
        """
        metric_folder = os.path.join(self.filename, metric)
        os.makedirs(metric_folder, exist_ok=True)

    def _save_metric(self, metric: str, series: pd.Series) -> None:
        """
        Function that saves the metrics in the respective folders
        If the file already exists, the new data is appended.
        """
        self._validate_folder(metric=metric)
        metric_filename = os.path.join(
            self.filename, metric, f"{self.uid}_{self.algorithm}.csv"
        )

        series.to_csv(
            metric_filename,
            mode="a",
            header=not os.path.exists(metric_filename),
            index=False,
        )
