import logging
import os

import numpy as np
import torch
import yaml
from torch import nn

from marloes.results.util import get_latest_uid

from .extractor import ExtensiveExtractor, Extractor


class Saver:
    """
    Class that saves the data extracted by the Extractor
    """

    def __init__(
        self, config: dict, test: bool = False, evaluate: bool = False
    ) -> None:
        self.name = "Saver"
        # allows testing with different filenames
        if test:
            self.base_file_path = "test"
        elif evaluate:
            self.base_file_path = "evaluate"
            scenario_name = config["data_config"].get("name")
            self.base_file_path = os.path.join(self.base_file_path, scenario_name)
        else:
            # default path for saving results
            self.base_file_path = "results"

        self.id_name = "uid.txt" if not test else "test_uid.txt"
        uid = config.get("uid", None)
        if uid:
            self.uid = uid
        elif not evaluate:
            self.uid = self._update_simulation_number()
        else:
            self.uid = get_latest_uid(self.base_file_path)
        self._save_config_to_yaml(config)

    def save(self, extractor: Extractor | ExtensiveExtractor) -> None:
        for (
            attr,
            value,
        ) in extractor.__dict__.items():  # loop over the attributes of the extractor
            if self._is_savable(value):
                self._save_metric(
                    attr, value[: extractor.i]
                )  # change to save only the data up to the current iteration

    def final_save(
        self, extractor: Extractor | ExtensiveExtractor, networks: list[nn.Module]
    ) -> None:
        """
        Should access the 'model' in the algorithm and save the weights/parameters into a file.
        In case of extensive extractor, the data from the results attribute should be saved here as well.
        """
        self.save(extractor)
        models_folder = os.path.join(self.base_file_path, "models")
        os.makedirs(models_folder, exist_ok=True)

        # Save the networks
        for network in networks:
            try:
                network_name = network.name
            except AttributeError:
                logging.warning(
                    f"Network {network} does not have a name attribute. Skipping saving."
                )
                continue
            network_folder = os.path.join(models_folder, network_name)
            os.makedirs(network_folder, exist_ok=True)
            network_filename = os.path.join(network_folder, f"{self.uid}.pt")
            torch.save(network.state_dict(), network_filename)

        # Save the results from the extensive extractor
        # First check if the extensive data attribute is present
        if isinstance(extractor, ExtensiveExtractor):
            dataframe_folder = self._validate_folder("dataframes")
            df = extractor.extensive_data.to_pandas()
            df.to_parquet(os.path.join(dataframe_folder, f"{self.uid}.parquet"))

    def _save_metric(self, metric: str, array: np.ndarray) -> None:
        """
        Function that saves the metrics in the respective folders
        If the file already exists, the new data is appended.
        """
        self._validate_folder(metric=metric)
        metric_filename = os.path.join(self.base_file_path, metric, f"{self.uid}.npy")
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
        config_filename = os.path.join(config_files, f"{self.uid}.yaml")
        with open(config_filename, "w") as f:
            yaml.dump(config, f)

    def _is_savable(self, value) -> bool:
        """
        Function that checks if the data is savable (numpy array)
        """
        return isinstance(value, np.ndarray)

    def _update_simulation_number(self) -> int:
        """
        Function that extracts and updates the uid.txt file in root folder/results/uid.txt
        """
        uid_path = os.path.join(self.base_file_path, self.id_name)
        with open(uid_path, "r+") as f:
            uid = int(f.read())
            f.seek(0)
            f.write(str(uid + 2))
            f.truncate()
        return uid

    def _validate_folder(self, metric: str) -> str:
        """
        Function that validates the folder for a single metric, if it does not exist, it is created
        """
        metric_folder = os.path.join(self.base_file_path, metric)
        os.makedirs(metric_folder, exist_ok=True)
        return metric_folder
