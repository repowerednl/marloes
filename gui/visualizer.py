from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QCheckBox,
    QPushButton,
    QScrollArea,
    QGroupBox,
    QFileDialog,
)
from PyQt6.QtCore import Qt

from gui.errors import ErrorScreen
from marloes.results.visualizer import Visualizer


class VisualizerGUI(QWidget):
    def __init__(self, metrics: list[str]):
        """
        Initialize the Visualizer GUI.
        """
        super().__init__()
        self.metrics = metrics

        self.setWindowTitle("Visualizer")
        self.setGeometry(100, 100, 500, 400)

        # Main layout
        layout = QVBoxLayout()

        # Simulation number input
        uid_layout = QHBoxLayout()
        uid_label = QLabel("Simulation UID:")
        self.uid_input = QLineEdit()
        uid_layout.addWidget(uid_label)
        uid_layout.addWidget(self.uid_input)
        layout.addLayout(uid_layout)

        # Metrics selection
        metrics_group = QGroupBox("Select Metrics to Visualize")
        metrics_layout = QVBoxLayout()
        self.metric_checkboxes = {}
        for metric in metrics:
            checkbox = QCheckBox(metric)
            self.metric_checkboxes[metric] = checkbox
            metrics_layout.addWidget(checkbox)
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)

        # Save PNG option
        self.save_pdf_checkbox = QCheckBox("Save graphs as PDF")
        layout.addWidget(self.save_pdf_checkbox)

        # Plot button
        self.plot_button = QPushButton("Plot")
        self.plot_button.clicked.connect(self.plot_metrics)
        layout.addWidget(self.plot_button)

        # Set main layout
        self.setLayout(layout)

    def plot_metrics(self):
        """
        Handle the Plot button click.
        """
        # Get the UID and selected metrics
        uids = self.uid_input.text()
        selected_metrics = [
            metric
            for metric, checkbox in self.metric_checkboxes.items()
            if checkbox.isChecked()
        ]
        save_pdf = self.save_pdf_checkbox.isChecked()

        if not selected_metrics:
            self.error_screen = ErrorScreen("Please select at least one metric.", self)
            self.error_screen.show()
            self.close()
            return

        if uids:
            uids = [int(uid.strip()) for uid in uids.split(",")]

        # Plot the metrics using the Visualizer
        Visualizer(uids).plot_metrics(selected_metrics, save_pdf)

        self.close()
