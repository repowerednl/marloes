from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import Qt


class ErrorScreen(QWidget):
    def __init__(self, error_message: str, parent_gui: QWidget = None):
        super().__init__()

        self.parent_gui = parent_gui  # Store the reference to the parent GUI

        # Set up the error screen
        self.setWindowTitle("Configuration Error")
        self.setGeometry(100, 100, 300, 150)
        self.setWindowModality(
            Qt.WindowModality.ApplicationModal
        )  # Block interaction with other windows

        # Layout
        layout = QVBoxLayout()

        # Error message
        error_label = QLabel(f"Error: {error_message}")
        error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(error_label)

        # Back button
        back_button = QPushButton("Back")
        back_button.clicked.connect(self.go_back)
        layout.addWidget(back_button)

        # Exit button
        exit_button = QPushButton("Exit")
        exit_button.clicked.connect(self.close)
        layout.addWidget(exit_button)

        # Set layout
        self.setLayout(layout)

    def go_back(self):
        """
        Close the error screen and show the parent GUI (if available).
        """
        self.close()
        if self.parent_gui:
            self.parent_gui.show()
