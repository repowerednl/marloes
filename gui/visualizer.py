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
    def __init__(self, visualizer: Visualizer, metrics: list[str]):
        """
        Initialize the Visualizer GUI.
        """
        super().__init__()
        self.visualizer = visualizer
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
        self.save_png_checkbox = QCheckBox("Save graphs as PNG")
        layout.addWidget(self.save_png_checkbox)

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
        uid = self.uid_input.text()
        selected_metrics = [
            metric
            for metric, checkbox in self.metric_checkboxes.items()
            if checkbox.isChecked()
        ]
        save_png = self.save_png_checkbox.isChecked()

        if not uid:
            ErrorScreen("Please enter a valid UID.")
            return

        if not selected_metrics:
            ErrorScreen("Please select at least one metric.")
            return

        # Plot the metrics using the Visualizer
        self.visualizer.plot_metric(uid, selected_metrics, save_png)
