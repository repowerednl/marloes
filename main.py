import sys
import argparse
import yaml
from PyQt6.QtWidgets import QApplication
import gui.startup as startup
from marloes.simulation import Simulation


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
    args = parser.parse_args()

    if args.default:
        config = load_config()
        for key, value in config.items():
            print(f"\n{key}: {value}")
        # TODO: run the experiment with the configuration
        simulation = Simulation(config)
        simulation.run()
    else:
        app = QApplication(sys.argv)
        window = startup.ExperimentSetupApp()
        window.show()
        sys.exit(app.exec())


if __name__ == "__main__":
    main()
