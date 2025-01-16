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
)
from PyQt6.QtCore import QTimer

from gui.success_screen import SuccessScreen
from src.marloes.algorithms import MADDPG, BaseAlgorithm, Priorities, SimpleSetpoint
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
        self.config_dropdown.addItems(["default_config", "simple_config"])
        layout.addWidget(self.config_dropdown)

        # ALGORITHM SELECTION
        self.algorithm_label = QLabel("Select Algorithm:")
        self.algorithm_dropdown = QComboBox(self)
        self.algorithm_dropdown.addItems(["Priorities", "SimpleSetpoint"])
        layout.addWidget(self.algorithm_label)
        layout.addWidget(self.algorithm_dropdown)

        # EXTRACTOR TYPE DROPDOWN
        self.extractor_type_label = QLabel("Extractor Type:")
        self.extractor_type_dropdown = QComboBox()
        self.extractor_type_dropdown.addItems(["default", "extensive"])
        layout.addWidget(self.extractor_type_label)
        layout.addWidget(self.extractor_type_dropdown)

        # EPOCHS
        self.epochs_label = QLabel("Epochs:")  # Must be an integer
        self.epochs = QSpinBox()
        self.epochs.setRange(1000, 1000000)
        self.epochs.setValue(100000)
        layout.addWidget(self.epochs_label)
        layout.addWidget(self.epochs)

        # Chunk size
        self.chunk_size_label = QLabel("Chunk Size:")  # Must be an integer
        self.chunk_size = QSpinBox()
        self.chunk_size.setRange(0, 1000000)
        self.chunk_size.setValue(10000)
        layout.addWidget(self.chunk_size_label)
        layout.addWidget(self.chunk_size)

        # PARAMETER INPUT
        self.learning_rate_label = QLabel("Learning Rate:")  # Must be a float
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setDecimals(5)
        self.learning_rate.setRange(0.00001, 0.1)
        self.learning_rate.setValue(0.001)
        self.learning_rate.setSingleStep(0.0001)
        self.learning_rate_label.hide()
        self.learning_rate.hide()
        layout.addWidget(self.learning_rate_label)
        layout.addWidget(self.learning_rate)

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

        if self.algorithm_dropdown.currentText():
            config["algorithm"] = self.algorithm_dropdown.currentText()

        if self.epochs.isVisible():
            config["epochs"] = self.epochs.value()

        if self.epochs.isVisible():
            config["chunk_size"] = self.chunk_size.value()

        if self.learning_rate.isVisible():
            config["learning_rate"] = self.learning_rate.value()

        config["extractor_type"] = self.extractor_type_dropdown.currentText()

        # TODO: add additional parameters as needed
        self.config = config

    def validate(self):
        try:
            e = validate_config(self.config)
        except ValueError as e:
            return False, str(e)
        return e == "Configuration is valid", e
