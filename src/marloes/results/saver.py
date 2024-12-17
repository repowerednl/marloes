import os
import yaml
import pandas as pd
import numpy as np

from .extractor import Extractor


class Saver:
    """
    Class that saves the data extracted by the Extractor
    """

    def __init__(self, config: dict) -> None:
        self.name = "Saver"
        self.algorithm = config["algorithm"]
        # allows testing with different filenames
        self.base_file_path = "results"
        self.id_name = "uid.txt"
        self.uid = self._update_simulation_number()
        self._save_config_to_yaml(config)

    def save(self, extractor: Extractor) -> None:
        for _, attr_value in extractor.__dict__.items():
            if self._is_savable(attr_value):
                for metric, array in attr_value.items():
                    self._save_metric(metric, array)

    def save_model(self, alg) -> None:
        """
        Should access the model in the algorithm and save the weights/parameters
        """
        pass

    def _save_metric(self, metric: str, array: np.ndarray) -> None:
        """
        Function that saves the metrics in the respective folders
        If the file already exists, the new data is appended.
        """
        self._validate_folder(metric=metric)
        metric_filename = os.path.join(
            self.base_file_path, metric, f"{self.uid}_{self.algorithm}.npy"
        )
        # save the data as .npy file, if it already exists, load the existing data and append the new data
        if os.path.exists(metric_filename):
            existing_data = np.load(metric_filename, mmap_mode="r+")
            array = np.concatenate((existing_data, array))
        np.save(metric_filename, array)

    def _save_config_to_yaml(self, config: dict) -> None:
        """
        Function that saves the configuration to a yaml file
        """
        config_files = os.path.join(self.base_file_path, "configs")
        os.makedirs(config_files, exist_ok=True)
        config_filename = os.path.join(
            config_files, f"{self.uid}_{self.algorithm}.yaml"
        )
        with open(config_filename, "w") as f:
            yaml.dump(config, f)

    def _is_savable(self, data: dict) -> bool:
        """
        Function that checks if the data is savable(should be a dictionary, with value as a np.array)
        """
        return isinstance(data, dict) and all(
            isinstance(value, np.ndarray) for value in data.values()
        )

    def _update_simulation_number(self) -> int:
        """
        Function that extracts and updates the uid.txt file in root folder/results/uid.txt
        """
        uid_path = os.path.join(self.base_file_path, self.id_name)
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
        metric_folder = os.path.join(self.base_file_path, metric)
        os.makedirs(metric_folder, exist_ok=True)
