import matplotlib.pyplot as plt
from marloes.results.calculator import Calculator


class Visualizer:
    def __init__(self, uid: int):
        """
        Initialize the Visualizer with a calculator instance.
        """
        self.calculator = Calculator(uid)
        self.uid = uid

    def plot_metric(self, metrics: list[str], save_png: bool = False):
        """
        Plots the selected metrics for the given simulation number.
        """
        # Pass the metrics to the calculator and get the calculated results
        metrics_data: dict = self.calculator.get_metrics(metrics)

        # Plot each metric
        for metric, data in metrics_data.items():
            print(f"Plotting {metric}...")
            if data is None:
                print(f"Metric {metric} could not be calculated.")
                continue

            plt.figure()
            plt.plot(data, label=metric)
            plt.title(f"{metric} - Simulation {self.uid}")
            plt.xlabel("Time")
            plt.ylabel(metric)
            plt.legend()

            if save_png:
                filename = f"results/images/{metric}_{self.uid}.png"
                plt.savefig(filename)
                print(f"Saved {metric} as {filename}")

            plt.show()
