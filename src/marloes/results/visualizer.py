import logging
import matplotlib.pyplot as plt
from marloes.results.calculator import Calculator


class Visualizer:
    def __init__(self, uids: list[int]):
        """
        Initialize the Visualizer with a list of UIDs.
        Each UID will have its own Calculator instance.
        """
        self.uids = uids

    def plot_metrics(self, metrics: list[str], save_png: bool = False):
        """
        Plots the selected metrics for each UID in a single figure per metric.
        """
        # If nothing is passed the uid must be extracted from the calculator
        if not self.uids:
            calculator = Calculator(self.uids)
            self.uids = [calculator.uid]
        logging.info(f"Plotting metrics {metrics} for simulations {self.uids}...")

        aggregated_data = {}

        # Retrieve the metrics data for each UID
        for uid in self.uids:
            calculator = Calculator(uid)
            logging.info(f"Getting metrics for UID {uid}...")
            metrics_data = calculator.get_metrics(metrics)

            # We remove or handle "info" so we don't accidentally try to plot it
            info_data = metrics_data.pop("info", None)
            calculator.log_sanity_check(info_data)

            # Organize data by metric
            for metric, data in metrics_data.items():
                if data is None:
                    logging.warning(
                        f"Metric '{metric}' for UID {uid} returned no data. Skipping..."
                    )
                    continue
                if metric not in aggregated_data:
                    aggregated_data[metric] = {}
                aggregated_data[metric][uid] = data

        plt.style.use("ggplot")

        # For each metric, create a single figure showing all UIDs (with some line style variation)
        line_styles = ["--", "-.", ":"]
        for metric, data_by_uid in aggregated_data.items():
            plt.figure(figsize=(10, 6))
            for uid, series in data_by_uid.items():
                plt.plot(
                    series,
                    label=f"UID {uid}",
                    alpha=0.8,
                    linestyle=line_styles[uid % len(line_styles)],
                )

            plt.title(
                f"{metric} across simulations {self.uids}",
                fontsize=14,
                fontweight="bold",
            )
            plt.xlabel("Time", fontsize=12)
            plt.ylabel(metric, fontsize=12)
            plt.legend(loc="best")
            plt.grid(True)

            if save_png:
                filename = (
                    f"results/images/{metric}_{'_'.join(map(str, self.uids))}.png"
                )
                plt.savefig(filename, dpi=300, bbox_inches="tight")
                logging.info(f"Saved {metric} as {filename}")

            plt.show()
