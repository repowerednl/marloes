import os
from PyQt6.QtWidgets import QLabel, QWidget, QVBoxLayout
from PyQt6.QtGui import QPixmap


def load_image(image_name: str) -> QPixmap:
    """Load an image from the static folder"""
    image_path = os.path.join(os.path.dirname(__file__), f"static/{image_name}")
    return QPixmap(image_path)


class LogoWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Logo Display")
        self.setGeometry(100, 100, 300, 300)

        layout = QVBoxLayout()

        logo_label = QLabel(self)
        logo_label.setPixmap(load_image("repowered-logo.png"))
        layout.addWidget(logo_label)

        self.setLayout(layout)
