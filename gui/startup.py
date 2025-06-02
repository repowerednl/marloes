from datetime import datetime, timedelta
import logging
import random
import time
from zoneinfo import ZoneInfo

import yaml
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
)
from PyQt6.QtCore import QTimer

from gui.success_screen import SuccessScreen
from gui.util import load_scenarios
from src.marloes.algorithms.dreamer import Dreamer
from src.marloes.algorithms.base import BaseAlgorithm
from src.marloes.algorithms.priorities import Priorities
from src.marloes.algorithms.simplesetpoint import SimpleSetpoint

from src.marloes.algorithms.dyna import Dyna
from src.marloes.validation.validate_config import validate_config

from .errors import ErrorScreen
from .img import LogoWindow
import os

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)


class ExperimentSetupApp(QWidget):
    def __init__(self):
        super().__init__()

        # Set up the window
        self.setWindowTitle("Experiment Setup")
        self.setGeometry(100, 100, 300, 200)

        # Layout
        layout = QVBoxLayout()

        # Add the Repowered logo
        self.logo = LogoWindow()
        layout.addWidget(self.logo)

        # UID input
        layout.addWidget(QLabel("UID (optional):"))
        self.uid_input = QLineEdit()
        self.uid_input.setPlaceholderText("Leave empty to skip")
        layout.addWidget(self.uid_input)

        # DEFAULT CONFIG CHECKBOX
        layout.addWidget(QLabel("Select configuration:"))
        self.config_dropdown = QComboBox()
        config_files = [
            f.replace(".yaml", "")
            for f in os.listdir("configs/")
            if f.endswith(".yaml")
        ]
        #config_files.sort(key=lambda x: "dreamer" not in x.lower())

        self.config_dropdown.addItems(config_files)
        layout.addWidget(self.config_dropdown)

        # SCENARIO DROPDOWN
        self.scenario_label = QLabel("Select Scenario:")
        self.scenario_dropdown = load_scenarios()
        layout.addWidget(self.scenario_label)
        layout.addWidget(self.scenario_dropdown)

        # ALGORITHM SELECTION
        self.algorithm_label = QLabel("Select Algorithm:")
        self.algorithm_dropdown = QComboBox(self)
        self.algorithm_dropdown.addItems(
            ["Dreamer", "Priorities", "SimpleSetpoint", "Dyna"]
        )
        layout.addWidget(self.algorithm_label)
        layout.addWidget(self.algorithm_dropdown)

        # EXTRACTOR TYPE DROPDOWN
        self.extractor_type_label = QLabel("Extractor Type:")
        self.extractor_type_dropdown = QComboBox()
        self.extractor_type_dropdown.addItems(["default", "extensive"])
        layout.addWidget(self.extractor_type_label)
        layout.addWidget(self.extractor_type_dropdown)

        # SUBREWARDS SELECTION WITH SCALING FACTORS
        self.subreward_group = QGroupBox("Select Subrewards:")
        self.subreward_layout = QVBoxLayout()

        self.subreward_checkboxes = {}
        self.subreward_scalings = {}

        for name in ["CO2", "SS", "NC", "NB", "NE"]:  # TODO: dynamically load this
            row = QHBoxLayout()

            checkbox = QCheckBox(name)
            checkbox.setChecked(name == "CO2")  # Default only CO2 selected
            self.subreward_checkboxes[name] = checkbox

            label = QLabel("Scaling:")
            scaling_box = QDoubleSpinBox()
            scaling_box.setRange(0.0, 1.0)
            scaling_box.setValue(1.0)
            scaling_box.setDecimals(2)
            scaling_box.setSingleStep(0.01)
            self.subreward_scalings[name] = scaling_box

            row.addWidget(checkbox)
            row.addWidget(label)
            row.addWidget(scaling_box)

            self.subreward_layout.addLayout(row)

        self.subreward_group.setLayout(self.subreward_layout)
        layout.addWidget(self.subreward_group)

        # EPOCHS
        self.training_steps_label = QLabel("Training Steps:")  # Must be an integer
        self.training_steps = QSpinBox()
        self.training_steps.setRange(1000, 1000000)
        self.training_steps.setValue(100000)
        layout.addWidget(self.training_steps_label)
        layout.addWidget(self.training_steps)

        # Chunk size
        self.chunk_size_label = QLabel("Chunk Size:")  # Must be an integer
        self.chunk_size = QSpinBox()
        self.chunk_size.setRange(0, 1000000)
        self.chunk_size.setValue(10000)
        layout.addWidget(self.chunk_size_label)
        layout.addWidget(self.chunk_size)

        # START BUTTON
        self.start_button = QPushButton("Start Experiment")
        self.start_button.clicked.connect(self.start_experiment)
        layout.addWidget(self.start_button)

        # Set layout
        self.setLayout(layout)

    def start_experiment(self):
        self.collect_config()

        algorithm: BaseAlgorithm = BaseAlgorithm.get_algorithm(
            self.config["algorithm"], self.config
        )
        start_time = time.time()
        try:
            algorithm.train()
        except Exception as e:
            self.close()
            raise e
        end_time = time.time()
        logging.info(f"Training took {end_time - start_time:.2f} seconds")

        # Show success message after training has finished
        self.success_screen = SuccessScreen()
        self.success_screen.show()
        self.close()

    def collect_config(self):
        config = {}
        # Load default config from YAML file
        chosen_config = self.config_dropdown.currentText()
        try:
            with open(f"configs/{chosen_config}.yaml", "r") as f:
                config = yaml.safe_load(f)
        except Exception as e:
            config = {}
            print(f"Error loading default config: {e}")

        # If UID is provided, add it to the config
        uid_text = self.uid_input.text().strip()
        uid = None
        if uid_text:
            try:
                uid = int(uid_text)
            except ValueError:
                logging.error("Invalid UID provided. Running new experiment.")
                pass

        if uid:
            passed_config = config.copy()
            try:
                with open(f"results/configs/{uid}.yaml", "r") as f:
                    config = yaml.safe_load(f)
            except FileNotFoundError:
                logging.error(f"Configuration file for UID {uid} not found.")
                return

            # Set start time to later to "continue" the training
            original_start_time = config.get("start_time")
            new_start_time = original_start_time + timedelta(
                minutes=config["training_steps"]
            )
            config["simulation_start_time"] = new_start_time
            config["uid"] = uid
            config["num_initial_random_steps"] = 0
            config["performed_training_steps"] = config["training_steps"]
            config["training_steps"] += passed_config["training_steps"]

        # Algorithm choice
        algorithm_choice = self.algorithm_dropdown.currentText()
        if algorithm_choice:
            if not config.get("algorithm") or algorithm_choice != "Priorities":
                config["algorithm"] = algorithm_choice

        # Scenario choice
        scenario = self.scenario_dropdown.currentText()
        try:
            with open(f"data_scenarios/{scenario}.yaml", "r") as f:
                scenario = yaml.safe_load(f)
        except Exception as e:
            logging.error(f"Could not load scenario: {e}")
            return

        # Update the config with the scenario
        config["data_config"] = scenario

        # Training steps
        if self.training_steps.isVisible():
            if (
                not config.get("training_steps")
                or self.training_steps.value() != 100000
            ):
                config["training_steps"] = self.training_steps.value()

        if self.chunk_size.isVisible():
            if not config.get("chunk_size") or self.chunk_size.value() != 10000:
                config["chunk_size"] = self.chunk_size.value()

        # Extractor type
        config["extractor_type"] = self.extractor_type_dropdown.currentText()

        # Subreward scaling factors
        selected_subrewards = {
            name: {
                "active": True,
                "scaling_factor": self.subreward_scalings[name].value(),
            }
            for name, checkbox in self.subreward_checkboxes.items()
            if checkbox.isChecked()
        }
        selected_subrewards.update(config.get("subrewards", {}))
        config["subrewards"] = selected_subrewards

        # Start time some minute in the first 4 months of 2025
        if "start_time" not in config:
            start_time = datetime(2025, 1, 1, tzinfo=ZoneInfo("UTC"))
            random_minutes = random.randint(0, 4 * 30 * 24 * 60)
            start_time += timedelta(minutes=random_minutes)
            config["start_time"] = start_time
            config["simulation_start_time"] = start_time

        self.config = config

    def validate(self):
        try:
            e = validate_config(self.config)
        except ValueError as e:
            return False, str(e)
        return e == "Configuration is valid", e
