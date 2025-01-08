from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import Qt


class SuccessScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Experiment Complete")
        self.setGeometry(100, 100, 300, 200)

        layout = QVBoxLayout()

        # Success message
        success_label = QLabel("The algorithm has finished training successfully!")
        success_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(success_label)

        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)

        self.setLayout(layout)
