from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QRadioButton,
    QPushButton,
    QLabel,
    QButtonGroup,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,  # Added import for QCheckBox
)

from .loading_screen import LoadingScreen
from .img import LogoWindow
from .errors import ErrorScreen
from src.marloes.validation.validate_config import validate_config
import yaml  # Import for handling YAML files


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

        # Label
        self.label = QLabel("Select experiment mode and setup:")
        layout.addWidget(self.label)

        # DEFAULT CONFIG CHECKBOX
        self.default_config_checkbox = QCheckBox("Use Default Config")
        layout.addWidget(self.default_config_checkbox)
        self.default_config_checkbox.toggled.connect(self.toggle_default_config)

        # GRID SEARCH
        self.grid_search_radio = QRadioButton("Grid Search")
        layout.addWidget(self.grid_search_radio)
        # Coverage input field if grid_search is selected
        self.coverage_label = QLabel("Coverage:")
        self.coverage_input = QSpinBox()
        self.coverage_input.setRange(1, 100)
        self.coverage_label.hide()
        self.coverage_input.hide()
        layout.addWidget(self.coverage_label)
        layout.addWidget(self.coverage_input)

        # ALGORITHM SELECTION
        self.algorithm_buttons = QButtonGroup(self)
        self.algorithm_model_based = QRadioButton("model_based")
        self.algorithm_model_free = QRadioButton("model_free")
        self.algorithm_simon_solver = QRadioButton("priorities")
        self.algorithm_buttons.addButton(self.algorithm_model_based)
        self.algorithm_buttons.addButton(self.algorithm_model_free)
        self.algorithm_buttons.addButton(self.algorithm_simon_solver)

        layout.addWidget(self.algorithm_model_based)
        layout.addWidget(self.algorithm_model_free)
        layout.addWidget(self.algorithm_simon_solver)

        # EPOCHS
        self.epochs_label = QLabel("Epochs:")  # Must be an integer
        self.epochs = QSpinBox()
        self.epochs.setRange(1000, 1000000)
        self.epochs.setValue(10000)
        layout.addWidget(self.epochs_label)
        layout.addWidget(self.epochs)

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

        # Toggle fields based on selections
        self.grid_search_radio.toggled.connect(self.toggle_grid_search)
        self.algorithm_buttons.buttonToggled.connect(self.toggle_params)
        self.default_config_checkbox.toggled.connect(self.toggle_default_config)

    def toggle_params(self):
        """Toggle parameters based on algorithm selection and grid search."""
        if self.default_config_checkbox.isChecked():
            self.hide_all_params()
            return

        selected_algorithm = self.algorithm_buttons.checkedButton()
        if selected_algorithm:
            algorithm = selected_algorithm.text()
            if algorithm == "priorities":
                self.hide_all_params()
            elif algorithm in ["model_based", "model_free"]:
                if not self.grid_search_radio.isChecked():
                    self.show_params()
                else:
                    self.hide_params()
        else:
            self.hide_all_params()

    def toggle_grid_search(self):
        """Toggle coverage input field and parameter fields based on grid search selection."""
        if self.default_config_checkbox.isChecked():
            self.hide_all_params()
            return

        if self.grid_search_radio.isChecked():
            self.coverage_label.show()
            self.coverage_input.show()
            self.hide_params()
        else:
            self.coverage_label.hide()
            self.coverage_input.hide()
            self.toggle_params()

    def toggle_default_config(self):
        """Toggle all input fields when default config is selected."""
        if self.default_config_checkbox.isChecked():
            # Hide all parameter fields
            self.hide_all_params()
            self.grid_search_radio.setChecked(False)
            self.grid_search_radio.setDisabled(True)
            # Deselect algorithms and disable them
            self.algorithm_model_based.setChecked(False)
            self.algorithm_model_free.setChecked(False)
            self.algorithm_simon_solver.setChecked(False)
            self.algorithm_buttons.setExclusive(False)
            self.algorithm_model_based.setDisabled(True)
            self.algorithm_model_free.setDisabled(True)
            self.algorithm_simon_solver.setDisabled(True)
            self.algorithm_buttons.setExclusive(True)
        else:
            # Enable all options again
            self.grid_search_radio.setDisabled(False)
            self.algorithm_model_based.setDisabled(False)
            self.algorithm_model_free.setDisabled(False)
            self.algorithm_simon_solver.setDisabled(False)

    def hide_params(self):
        """Hides parameter fields (not needed for grid search or priorities)."""
        self.learning_rate_label.hide()
        self.learning_rate.hide()

    def show_params(self):
        """Shows parameter fields (needed for model_based and model_free)."""
        self.learning_rate_label.show()
        self.learning_rate.show()

    def hide_all_params(self):
        """Hides all input fields when default config is selected."""
        self.epochs_label.hide()
        self.epochs.hide()
        self.coverage_label.hide()
        self.coverage_input.hide()
        self.hide_params()

    def start_experiment(self):
        self.collect_config()
        valid, error_message = self.validate()
        if not valid:
            self.error_screen = ErrorScreen(error_message, self)
            self.error_screen.show()
            self.close()
        else:
            print(f"Running experiment with configuration: {self.config}")
            # Switch to the loading screen
            self.loading_screen = LoadingScreen(self.config)
            self.loading_screen.show()
            self.close()

    def collect_config(self):
        config = {}
        if self.default_config_checkbox.isChecked():
            # Load default config from YAML file
            try:
                with open("configs/default_config.yaml", "r") as f:
                    config = yaml.safe_load(f)
            except Exception as e:
                config = {}
                print(f"Error loading default config: {e}")
        else:
            # Collect user inputs
            config["grid_search"] = (
                True if self.grid_search_radio.isChecked() else False
            )
            config["coverage"] = (
                self.coverage_input.value()
                if self.grid_search_radio.isChecked()
                else None
            )
            config["algorithm"] = (
                self.algorithm_buttons.checkedButton().text()
                if self.algorithm_buttons.checkedButton()
                else None
            )
            config["epochs"] = self.epochs.value() if self.epochs.isVisible() else None
            config["learning_rate"] = (
                self.learning_rate.value() if self.learning_rate.isVisible() else None
            )
            # TODO: add additional parameters as needed
        self.config = config

    def validate(self):
        try:
            e = validate_config(self.config)
        except ValueError as e:
            return False, str(e)
        return e == "Configuration is valid", e
