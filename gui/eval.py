from datetime import datetime
import os
import glob
import logging
from zoneinfo import ZoneInfo

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QComboBox,
    QLineEdit,
    QPushButton,
    QSpinBox,
)

from gui.img import LogoWindow
from gui.util import load_scenarios
from marloes.results.util import get_latest_uid
from src.marloes.algorithms.base import BaseAlgorithm
import yaml

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)


class EvaluationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Evaluate Experiment")
        self.setGeometry(100, 100, 300, 200)

        layout = QVBoxLayout()

        # Add logo
        self.logo = LogoWindow()
        layout.addWidget(self.logo)

        # UID input
        layout.addWidget(QLabel("UID (optional):"))
        self.uid_input = QLineEdit()
        self.uid_input.setPlaceholderText("Leave empty to take latest UID")
        layout.addWidget(self.uid_input)

        # Scenario dropdown from configs/*.yaml
        layout.addWidget(QLabel("Select scenario:"))
        self.scenario_dropdown = load_scenarios()
        layout.addWidget(self.scenario_dropdown)

        # Number of evaluation steps
        layout.addWidget(QLabel("Number of evaluation steps:"))
        self.eval_steps_input = QSpinBox()
        self.eval_steps_input.setRange(1, 1000000)
        self.eval_steps_input.setSingleStep(100)
        self.eval_steps_input.setValue(15000)
        layout.addWidget(self.eval_steps_input)

        # Start button
        self.start_button = QPushButton("Start Evaluation")
        self.start_button.clicked.connect(self.start_evaluation)
        layout.addWidget(self.start_button)

        self.setLayout(layout)

    def start_evaluation(self):
        uid_text = self.uid_input.text().strip()
        uid = int(uid_text) if uid_text.isdigit() else None
        scenario = self.scenario_dropdown.currentText()

        if not uid:
            uid = get_latest_uid("results")

        try:
            with open(f"results/configs/{uid}.yaml", "r") as f:
                config: dict = yaml.safe_load(f)
        except FileNotFoundError:
            logging.error(f"Configuration file for UID {uid} not found.")
            return

        try:
            with open(f"data_scenarios/{scenario}.yaml", "r") as f:
                scenario = yaml.safe_load(f)
        except Exception as e:
            logging.error(f"Could not load scenario: {e}")
            return

        # Delete previous evalation files with this UID
        clear_all_files_with_uid(uid, scenario["name"])

        # Update the config with the scenario
        config["data_config"] = scenario

        # Update the config with the number of evaluation steps
        config["eval_steps"] = self.eval_steps_input.value()

        # Start time is the first of september 2025 for evaluation
        start_time = datetime(2025, 9, 1, tzinfo=ZoneInfo("UTC"))
        config["simulation_start_time"] = start_time
        config["start_time"] = start_time
        config["uid"] = uid

        print(f"Loaded config: {config}")

        logging.info(f"Evaluating scenario '{scenario}' with UID: {uid}")
        algorithm: BaseAlgorithm = BaseAlgorithm.get_algorithm(
            config["algorithm"], config, evaluate=True
        )

        try:
            algorithm.eval()
        except Exception as e:
            logging.error(f"Error during evaluation: {e}")
            raise

        logging.info("Evaluation completed.")
        self.close()


def clear_all_files_with_uid(uid: int, scenario: str) -> None:
    """
    Clear all files related to a specific UID.
    Should recursively go through "evaluate" folder and subfolders and delete all files
    that match the UID in their filename.
    """
    path = f"evaluate/{scenario}/"
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if str(uid) in filename:
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    logging.info(f"Deleted file: {file_path}")
                except Exception as e:
                    logging.error(f"Error deleting file {file_path}: {e}")
            else:
                logging.debug(f"Skipping file: {filename} (does not match UID {uid})")
