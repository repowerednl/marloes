import sys
import argparse
import numpy as np
import yaml
from PyQt6.QtWidgets import QApplication
import gui.startup as startup
from gui.visualizer import VisualizerGUI
from marloes.results.calculator import Calculator
from marloes.data.metrics import Metrics
from marloes.results.visualizer import Visualizer
from marloes.marloes import MARLOES


def load_config():
    with open("configs/default_config.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config


def parse_arguments():
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
    parser.add_argument(
        "--seed",
        type=int,
        help="Set the seed for the random number generator",
    )
    return parser.parse_args()


def run_default_mode():
    config = load_config()
    for key, value in config.items():
        print(f"\n{key}: {value}")
    # algorithm = Algorithm(config)
    # algorithm.train()


def run_app_mode(args):
    np.random.seed(args.seed if args.seed else 42)
    app = QApplication(sys.argv)

    if args.visualizer:
        available_metrics = [item.value for item in Metrics]
        visualizer_window = VisualizerGUI(available_metrics)
        visualizer_window.show()
    else:
        window = startup.ExperimentSetupApp()
        window.show()

    sys.exit(app.exec())


def main():
    print(MARLOES)
    args = parse_arguments()

    if args.default:
        run_default_mode()
    else:
        run_app_mode(args)


if __name__ == "__main__":
    main()
