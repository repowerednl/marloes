import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from marloes.results.calculator import Calculator
from marloes.results.metrics import Metrics
import plotly.graph_objects as go
from PIL import Image


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

        for metric, data_by_uid in aggregated_data.items():
            if metric == Metrics.EXTENSIVE_DATA:
                for uid, df in data_by_uid.items():
                    self.plot_sankey(df, uid, save_png)
            else:
                self.plot_default(metric, data_by_uid, save_png)

    def plot_default(
        self, metric: str, data_by_uid: dict[int, np.ndarray], save_png: bool = False
    ):
        """
        Plots the default metrics for each UID in a single figure per metric.
        """
        plt.style.use("ggplot")

        # For each metric, create a single figure showing all UIDs (with some line style variation)
        line_styles = ["--", "-.", ":"]
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
            filename = f"results/images/{metric}_{'_'.join(map(str, self.uids))}.png"
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            logging.info(f"Saved {metric} as {filename}")

        plt.show()

    def plot_sankey(self, df: pd.DataFrame, uid: int, save_png: bool = False):
        """
        Generate a Sankey diagram from a DataFrame with flow information.
        """
        df = df.filter(like="_to_").round(0)

        # Extract unique nodes
        nodes = list(
            set(col.split("_to_")[0] for col in df.columns)
            | set(col.split("_to_")[1] for col in df.columns)
        )

        # Create node index mappings
        node_indices = {node: idx for idx, node in enumerate(nodes)}

        sources, targets, values = [], [], []
        node_colors = []

        # TODO: Color the nodes appropriately
        for node in nodes:
            node_colors.append(get_node_color(node))

        # Process the links/flows
        link_colors = []
        for col in df.columns:
            source, target = col.split("_to_")
            sources.append(node_indices[source])
            targets.append(node_indices[target])
            flow_value = df[col].sum()
            values.append(flow_value)

            # Link color should match the source node's color
            link_colors.append(get_node_color(source))

        # Generate the sankey
        sankey_fig = go.Figure(
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=nodes,
                    color=node_colors,
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=link_colors,
                ),
            )
        )

        # Add title
        title = "Energy Flows"
        sankey_fig.update_layout(title_text=title, font_size=10)
        sankey_fig.show()


# Define colors for the nodes and flows
colors = {
    "logogreen": "#13A538",
    "darkblue": "#244546",
    "darkerblue": "#162A2A",
    "oceanblue": "#0093D3",
    "mintgreen": "#3F4E55",
    "greyblue": "#EAF6FE",
    "grey": "#C8D3D9",
    "darkgrey": "#627076",
    "red": "#D61010",
    "palered": "#E97171",
    "orange": "#FF6E00",
    "paleorange": "#FFF0E1",
    "yellow": "#FFC700",
    "paleyellow": "#FFF4CC",
    "palegreen": "#7ACA8E",
    "palegrey": "#C8D3D9",
    "paleblue": "#75C5E8",
}


def get_node_color(node_name):
    if "Solar" in node_name:
        return colors["oceanblue"]
    elif "Grid" in node_name:
        return colors["grey"]
    elif "Demand" in node_name:
        return colors["greyblue"]
    elif "Battery" in node_name:
        return colors["logogreen"]
    else:
        return colors["mintgreen"]


def get_flow_color(node_name):
    # TODO: This does not work yet
    # Pale versions of the node colors
    if "Solar" in node_name:
        return colors["red"]
    elif "Grid" in node_name:
        return colors["palegrey"]
    elif "Battery" in node_name:
        return colors["palegreen"]
