import logging

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
)
from PyQt6.QtCore import QTimer

from gui.success_screen import SuccessScreen
from src.marloes.algorithms import BaseAlgorithm, Priorities, SimpleSetpoint, Dreamer
from src.marloes.algorithms.dyna import Dyna
from src.marloes.validation.validate_config import validate_config

from .errors import ErrorScreen
from .img import LogoWindow

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

        # DEFAULT CONFIG CHECKBOX
        layout.addWidget(QLabel("Select configuration:"))
        self.config_dropdown = QComboBox()
        self.config_dropdown.addItems(
            ["default_config", "simple_config", "dyna_config", "test_config"]
        )
        layout.addWidget(self.config_dropdown)

        # ALGORITHM SELECTION
        self.algorithm_label = QLabel("Select Algorithm:")
        self.algorithm_dropdown = QComboBox(self)
        self.algorithm_dropdown.addItems(
            ["Priorities", "SimpleSetpoint", "Dyna", "Dreamer"]
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

        for name in ["CO2", "SS", "NC", "NB", "NE"]:
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
        # valid, error_message = self.validate()
        # if not valid:
        #     self.error_screen = ErrorScreen(error_message, self)
        #     self.error_screen.show()
        #     self.close()
        # else:
        logging.info("Starting experiment with the following configuration:")
        for key, value in self.config.items():
            logging.info(f"     {key}: {value}")
        # Switch to the loading screen
        # self.loading_screen = LoadingScreen(self.config)
        # self.loading_screen.show()
        # self.close()
        algorithm: BaseAlgorithm = BaseAlgorithm.get_algorithm(
            self.config["algorithm"], self.config
        )
        try:
            algorithm.train()
        except Exception as e:
            # print(f"Error during training: {e}")
            # self.error_screen = ErrorScreen("Error during training", self)
            # self.error_screen.show()
            self.close()
            raise e

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

        algorithm_choice = self.algorithm_dropdown.currentText()
        if algorithm_choice:
            if not config.get("algorithm") or algorithm_choice != "Priorities":
                config["algorithm"] = algorithm_choice

        if self.training_steps.isVisible():
            if (
                not config.get("training_steps")
                or self.training_steps.value() != 100000
            ):
                config["training_steps"] = self.training_steps.value()

        if self.chunk_size.isVisible():
            if not config.get("chunk_size") or self.chunk_size.value() != 10000:
                config["chunk_size"] = self.chunk_size.value()

        config["extractor_type"] = self.extractor_type_dropdown.currentText()

        selected_subrewards = {
            name: {
                "active": True,
                "scaling_factor": self.subreward_scalings[name].value(),
            }
            for name, checkbox in self.subreward_checkboxes.items()
            if checkbox.isChecked()
        }
        combined_rewards = selected_subrewards.update(config.get("subrewards", {}))
        config["subrewards"] = combined_rewards

        # TODO: add additional parameters as needed
        self.config = config

    def validate(self):
        try:
            e = validate_config(self.config)
        except ValueError as e:
            return False, str(e)
        return e == "Configuration is valid", e
