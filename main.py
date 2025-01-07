import sys
import argparse
import yaml
from PyQt6.QtWidgets import QApplication
import gui.startup as startup
from gui.visualizer import VisualizerGUI
from marloes.results.calculator import Calculator
from marloes.results.metrics import Metrics
from marloes.results.visualizer import Visualizer


def load_config():
    with open("configs/default_config.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--default",
        action="store_true",
        help="Load config file instead of running the app",
    )
    parser.add_argument(
        "--visualizer",
        action="store_true",
        help="Start the Visualizer GUI",
    )
    args = parser.parse_args()

    if args.default:
        # Load the default configuration and run the simulation
        config = load_config()
        for key, value in config.items():
            print(f"\n{key}: {value}")
        # algorithm = Algorithm(config)
        # algorithm.train()
    else:
        app = QApplication(sys.argv)

        if args.visualizer:
            # Define available metrics
            available_metrics = [item.value for item in Metrics]

            # Show the Visualizer GUI
            visualizer_window = VisualizerGUI(available_metrics)
            visualizer_window.show()
        else:
            # Start the normal ExperimentSetupApp GUI
            window = startup.ExperimentSetupApp()
            window.show()

        sys.exit(app.exec())


if __name__ == "__main__":
    main()
