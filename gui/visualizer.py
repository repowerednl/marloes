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
    def __init__(self):
        """
        Initialize the Visualizer GUI.
        """
        super().__init__()

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

        # Button to load metrics into the group
        self.load_metrics_button = QPushButton("Load Metrics")
        self.load_metrics_button.clicked.connect(self.load_metrics)
        layout.addWidget(self.load_metrics_button)

        # Metrics selection (empty initially)
        self.metrics_group = QGroupBox("Select Metrics to Visualize")
        self.metrics_layout = QVBoxLayout()
        self.metric_checkboxes = {}
        self.metrics_group.setLayout(self.metrics_layout)
        layout.addWidget(self.metrics_group)

        # Apply rolling median option
        self.apply_rolling_median_checkbox = QCheckBox("Apply rolling median")
        self.apply_rolling_median_checkbox.setChecked(True)
        layout.addWidget(self.apply_rolling_median_checkbox)

        # Save PNG option
        self.save_png_checkbox = QCheckBox("Save graphs as PNG")
        layout.addWidget(self.save_png_checkbox)

        # Plot button
        self.plot_button = QPushButton("Plot")
        self.plot_button.clicked.connect(self.plot_metrics)
        layout.addWidget(self.plot_button)

        # Set main layout
        self.setLayout(layout)

    def get_uids(self):
        """
        Get the UIDs from the input field.
        """
        uids = self.uid_input.text()
        if not uids:
            return []

        # Split by comma and convert to integers
        uids = [int(uid.strip()) for uid in uids.split(",")]
        return uids

    def load_metrics(self):
        """
        Populate the metrics group with checkboxes for each metric in self.metrics.
        """
        # Get uids to initialize the visualizer
        uids = self.get_uids()
        self.visualizer = Visualizer(uids)
        metrics = self.visualizer.get_common_metrics()

        if not metrics:
            metrics = ["No metrics found."]

        # Clear old checkboxes if 'Load Metrics' is pressed more than once
        for i in reversed(range(self.metrics_layout.count())):
            widget = self.metrics_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        self.metric_checkboxes.clear()

        # Create and add a checkbox for each metric
        for metric in metrics:
            checkbox = QCheckBox(metric)
            self.metric_checkboxes[metric] = checkbox
            self.metrics_layout.addWidget(checkbox)

    def plot_metrics(self):
        """
        Handle the Plot button click.
        """
        # Get the UID(s) and selected metrics
        selected_metrics = [
            metric
            for metric, checkbox in self.metric_checkboxes.items()
            if checkbox.isChecked()
        ]
        save_png = self.save_png_checkbox.isChecked()
        rolling_median = self.apply_rolling_median_checkbox.isChecked()

        if not selected_metrics:
            self.error_screen = ErrorScreen("Please select at least one metric.", self)
            self.error_screen.show()
            self.close()
            return

        # Plot the metrics using the Visualizer
        self.visualizer.plot_metrics(selected_metrics, save_png, rolling_median)
        self.close()
