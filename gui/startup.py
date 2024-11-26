from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QRadioButton,
    QPushButton,
    QLabel,
    QButtonGroup,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
)

from .loading_screen import LoadingScreen
from .img import LogoWindow
from .errors import ErrorScreen
from src.marloes.validation.validate_config import validate_config

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
        self.algorithm_buttons.addButton(self.algorithm_model_based)
        self.algorithm_buttons.addButton(self.algorithm_model_free)

        layout.addWidget(self.algorithm_model_based)
        layout.addWidget(self.algorithm_model_free)

        # EPOCHS
        self.epochs_label = QLabel("Epochs:") # must be an integer
        self.epochs = QSpinBox()
        self.epochs.setRange(1000, 1000000)
        self.epochs.setValue(10000)
        layout.addWidget(self.epochs_label)
        layout.addWidget(self.epochs)

        # PARAMETER INPUT
        self.learning_rate_label = QLabel("Learning Rate:") # must be a float
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

    def toggle_params(self):
        """ toggle parameters if grid_search is not checked, and either model_based or model_free is checked """
        if not self.grid_search_radio.isChecked() and (self.algorithm_model_based.isChecked() or self.algorithm_model_free.isChecked()):
            self.show_params()
        else:
            self.hide_params()

    def hide_params(self):
        """ Hides all parameter fields (not needed for grid_search)"""
        self.learning_rate_label.hide()
        self.learning_rate.hide()

    def show_params(self):
        """ Shows all parameter fields (needed for model_based and model_free)"""
        self.learning_rate_label.show()
        self.learning_rate.show()

    def toggle_grid_search(self):
        """ toggle coverage input field and hide params if grid_search is checked """
        if self.grid_search_radio.isChecked():
            self.coverage_label.show()
            self.coverage_input.show()
            self.hide_params()
        else:
            self.coverage_label.hide()
            self.coverage_input.hide()
            self.show_params()

    def start_experiment(self):
        self.collect_config()
        valid, e = self.validate()
        # also collect all other information from the user interface
        if not valid:
            self.error_screen = ErrorScreen(e)
            self.error_screen.show()
            self.close()
        else:
            print(f"Running {'Grid Search ' if self.config['grid_search'] else 'with '} {self.config['algorithm']} Algorithm...")
            print(f"Epochs: {self.config['epochs']}")
            
            # add the logic for each experiment mode, here or in probably in loading screen
            # Example: self.run_grid_search() or self.run_algorithm()

            # Switch to the loading screen
            self.loading_screen = LoadingScreen(self.config)
            self.loading_screen.show()
            self.close()
    
    def collect_config(self):
        config = {}
        # Determine which option is selected
        config['grid_search'] = True if self.grid_search_radio.isChecked() else False
        config['coverage'] = self.coverage_input.text() if self.grid_search_radio.isChecked() else None
        config['algorithm'] = self.algorithm_buttons.checkedButton().text() if self.algorithm_buttons.checkedButton() else None
        config['epochs'] = self.epochs.value() 
        config['learning_rate'] = self.learning_rate.value() if not self.grid_search_radio.isChecked() else None
        # TODO: add params
        
        self.config = config

    def validate(self):
        e = validate_config(self.config)
        return e == "Configuration is valid", e