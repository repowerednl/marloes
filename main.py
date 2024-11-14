import sys

from PyQt6.QtWidgets import QApplication
import gui.startup as startup

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    window = startup.ExperimentSetupApp()
    window.show()
    
    sys.exit(app.exec())