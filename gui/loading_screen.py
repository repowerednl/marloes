from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar
from PyQt6.QtCore import Qt, QTimer
from .img import LogoWindow

class LoadingScreen(QWidget):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        self.validate(config)
        
        # Set up the loading screen
        self.setWindowTitle("Loading Experiment")
        self.setGeometry(100, 100, 300, 100)
        
        # Layout
        layout = QVBoxLayout()

        # Add the Repowered logo
        self.logo = LogoWindow()
        layout.addWidget(self.logo)

        # show the configuration elements
        for key, value in self.config.items():
            label = QLabel(f"{key}: {value}")
            layout.addWidget(label)
        
        # Loading label
        self.loading_label = QLabel("Experiment is running, please wait...")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.loading_label)
        
        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)
        
        # Set layout
        self.setLayout(layout)
        
        # Start a timer to simulate loading progress
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(50)  # Update every 50 ms

        self.progress_value = 0

    def update_progress(self):
        if self.progress_value < 100:
            self.progress_value += 1
            self.progress_bar.setValue(self.progress_value)
        else:
            # Stop timer when loading is complete
            self.timer.stop()
            # Here, add code to proceed to the experiment or main screen
            self.loading_label.setText("Experiment Finished!")
            # Optionally close loading screen or transition to the next screen