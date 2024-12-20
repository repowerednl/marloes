import matplotlib.pyplot as plt


class Visualizer:
    def __init__(self, calculator):
        """
        Initialize the Visualizer with a calculator instance.
        """
        self.calculator = calculator

    def plot_metric(self, uid: str, metrics: list[str], save_png: bool = False):
        """
        Plots the selected metrics for the given simulation number.
        """
        # Pass the metrics to the calculator and get the calculated results
        metrics_data = {metric: None for metric in metrics}
        metrics_data = self.calculator.calculate(uid, metrics_data)

        # Plot each metric
        for metric, data in metrics_data.items():
            if data is None:
                print(f"Metric {metric} could not be calculated.")
                continue

            plt.figure()
            plt.plot(data, label=metric)
            plt.title(f"{metric} - Simulation {uid}")
            plt.xlabel("Time")
            plt.ylabel(metric)
            plt.legend()

            if save_png:
                filename = f"{metric}_{uid}.png"
                plt.savefig(filename)
                print(f"Saved {metric} as {filename}")

            plt.show()
