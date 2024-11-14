from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import Qt

class ErrorScreen(QWidget):
    def __init__(self, error_message: str):
        super().__init__()

        # Set up the error screen
        self.setWindowTitle("Configuration Error")
        self.setGeometry(100, 100, 300, 150)
        
        # Layout
        layout = QVBoxLayout()
        
        # Error message
        error_label = QLabel(f"Error: {error_message}")
        error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(error_label)
        
        # Back button
        back_button = QPushButton("Back to Experiment Setup")
        back_button.clicked.connect(self.go_back)
        layout.addWidget(back_button)

        # Exit button
        exit_button = QPushButton("Exit")
        exit_button.clicked.connect(self.close)
        layout.addWidget(exit_button)
        
        # Set layout
        self.setLayout(layout)

    def go_back(self):
        # Close the error screen
        self.close()
        # Show the experiment setup screen
        from .startup import ExperimentSetupApp
        self.experiment_app = ExperimentSetupApp()
        self.experiment_app.show()



