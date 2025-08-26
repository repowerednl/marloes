import argparse
import os
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from copy import deepcopy
from gui.eval import clear_all_files_with_uid
from marloes.agents.base import Agent
from src.marloes.algorithms.dreamer import Dreamer
from src.marloes.algorithms.base import BaseAlgorithm
from src.marloes.algorithms.priorities import Priorities
from src.marloes.algorithms.simplesetpoint import SimpleSetpoint
from src.marloes.algorithms.dyna import Dyna
from src.marloes.results.visualizer import Visualizer

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import hex_to_rgb

import time
from scipy.stats import sem, iqr, median_abs_deviation
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar

import logging

X = pd.date_range(start="2025-09-01", periods=15000, freq="min")


def plot_losses(
    base: str,
    configs: list[str] = ["3_handler", "6_handler", "9_handler"],
    metrics: dict = None,
    save: bool = True,
):
    """
    This function gathers results from multiple configs and plots them in two figures:
    - world_model_losses_fig: Dynamics, Prediction and Representation losses (1x3)
    - actor_critic_fig: Critic and Actor losses (1x2)
    It also computes the AULC (area under the mean loss curve) for each config & metric,
    saves them to a DataFrame, and prints the DataFrame at the end.
    """
    color_map = {
        configs[0]: "#f79244",  # orange
        configs[1]: "#44a9f7",  # blue
        configs[2]: "#202C59",  # dark blue
    }
    if not metrics:
        metrics = {
            "Dynamics Loss": "dynamics_loss",
            "Prediction Loss": "prediction_loss",
            "Representation Loss": "representation_loss",
            "Critic Loss": "critic_loss",
            "Actor Loss": "actor_loss",
        }
    root = "results"
    experiment_dir = os.path.join(root, "experiment")

    # load all data from the given metrics for each config
    all_data = {}
    for config in configs:
        df = pd.read_csv(os.path.join(experiment_dir, base + config + ".csv"))
        uids = df["uid"].iloc[:-1].tolist()
        all_data[config] = {}
        for metric_name, metric_key in metrics.items():
            metric_data = []
            for uid in uids:
                metric_file = os.path.join(root, metric_key, f"{uid}.npy")
                if os.path.exists(metric_file):
                    metric_data.append(np.load(metric_file))
                else:
                    logging.warning(f"Warning: {metric_file} does not exist.")
            if metric_data:
                all_data[config][metric_name] = np.array(metric_data)
            else:
                logging.warning(
                    f"Warning: No data found for metric {metric_name} in config {config}."
                )

    # Prepare subplots
    world_model_metrics = ["Dynamics Loss", "Prediction Loss", "Representation Loss"]
    actor_critic_metrics = ["Critic Loss", "Actor Loss"]

    world_model_losses_fig = make_subplots(
        rows=1, cols=3, subplot_titles=world_model_metrics
    )
    actor_critic_fig = make_subplots(
        rows=1, cols=2, subplot_titles=actor_critic_metrics
    )

    # storage for AULC values
    aulc_records = []

    # Helper to process, plot and compute AULC for a single metric
    def plot_metric(fig, row, col, metric_name, showlegend=True):
        for config in configs:
            if metric_name not in all_data[config]:
                continue
            col_hex = color_map[config]
            # load and clean
            metric_values = all_data[config][metric_name]
            low = np.percentile(metric_values, 2.5, axis=0)
            high = np.percentile(metric_values, 97.5, axis=0)
            arr = np.where(
                (metric_values >= low) & (metric_values <= high), metric_values, np.nan
            )
            df_arr = pd.DataFrame(arr).ffill().bfill().values
            avg = np.nanmean(df_arr, axis=0)
            std = sem(df_arr, axis=0, nan_policy="omit")
            valid = ~np.isnan(avg)
            avg = avg[valid]
            std = std[valid]
            # smoothing
            avg = pd.Series(avg).rolling(window=50, min_periods=1).mean().values
            std = pd.Series(std).rolling(window=50, min_periods=1).mean().values
            steps = np.arange(len(avg))

            # compute AULC via trapezoidal rule
            aulc = np.trapz(avg, x=steps)
            aulc_records.append({"config": config, "metric": metric_name, "AULC": aulc})

            # ribbons & line
            r, g, b = hex_to_rgb(col_hex)
            upper = avg + std
            lower = avg - std
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=upper,
                    line=dict(width=0),
                    fillcolor=f"rgba({r},{g},{b},0.2)",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=lower,
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=f"rgba({r},{g},{b},0.2)",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )
            name = config.replace("_", "-")
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=avg,
                    mode="lines",
                    name=name,
                    line=dict(width=2, color=col_hex),
                    showlegend=showlegend,
                ),
                row=row,
                col=col,
            )

    # Plot and compute AULC for world-model losses
    for idx, m in enumerate(world_model_metrics, start=1):
        plot_metric(world_model_losses_fig, 1, idx, m, (idx == 1))
    world_model_losses_fig.update_layout(
        title="World‐Model Loss during training", template="plotly_white"
    )
    world_model_losses_fig.update_xaxes(title_text="Steps", row=1, col=2)
    world_model_losses_fig.update_yaxes(title_text="Value", row=1, col=1)
    world_model_losses_fig.show()
    if save:
        world_model_losses_fig.write_image(os.path.join(root, "world_model_losses.png"))

    # Plot and compute AULC for actor & critic losses
    for idx, m in enumerate(actor_critic_metrics, start=1):
        plot_metric(actor_critic_fig, 1, idx, m, (idx == 1))
    actor_critic_fig.update_layout(
        title="Actor & Critic Loss during training", template="plotly_white"
    )
    actor_critic_fig.update_xaxes(title_text="Steps", row=1, col=1)
    actor_critic_fig.update_xaxes(title_text="Steps", row=1, col=2)
    actor_critic_fig.update_yaxes(title_text="Value", row=1, col=1)
    actor_critic_fig.update_yaxes(title_text="Value", row=1, col=2)
    actor_critic_fig.show()
    if save:
        actor_critic_fig.write_image(os.path.join(root, "actor_critic_losses.png"))

    # build and print AULC DataFrame
    aulc_df = pd.DataFrame(aulc_records)
    aulc_df = aulc_df.pivot(index="metric", columns="config", values="AULC")
    print("\nArea Under Learning Curve (mean loss) for each config & metric:\n")
    print(aulc_df)


def plot_noise(
    base: str,
    configs: list[str] = ["3_handler", "3_handler_default"],
    metrics: dict = {
        "Reward": "reward",
        "Grid production": "total_grid_production",
    },
    save: bool = False,
):
    """
    This function gathers results from given configs and plots the metrics with mean and std in one figure.
    """
    root = "results"
    experiment_dir = os.path.join(root, "experiment")
    # Gathering data
    all_data = {}
    for config in configs:
        df = pd.read_csv(os.path.join(experiment_dir, base + config + ".csv"))
        uids = df["uid"].iloc[:-1].tolist()
        all_data[config] = {}
        for metric_name, metric_key in metrics.items():
            metric_data = []
            for uid in uids:
                metric_file = os.path.join(root, metric_key, f"{uid}.npy")
                if os.path.exists(metric_file):
                    data = np.load(metric_file)
                    metric_data.append(data)
                else:
                    logging.warning(f"Warning: {metric_file} does not exist.")
                if metric_data:
                    all_data[config][metric_name] = np.array(metric_data)
                else:
                    logging.warning(
                        f"Warning: No data found for metric {metric_name} in config {config}."
                    )

    # Plotting the data, make 1 row, 2 columns. Left should be cumulative reward for both configs, right should be cumulative grid production for both configs
    fig = make_subplots(rows=1, cols=2, subplot_titles=list(metrics.keys()))
    showlegend = 0
    for metric_name, metric_key in metrics.items():
        for config, data in all_data.items():
            if metric_name in data:
                if config == "3_handler_default":
                    legend = "with noise"
                    # color should be red (in hex)
                    col = "#d62728"
                    r, g, b = hex_to_rgb(col)
                if config == "3_handler":
                    legend = "without noise"
                    # color should be blue
                    col = "#1f77b4"
                    r, g, b = hex_to_rgb(col)
                metric_values = data[metric_name]
                avg_values = np.mean(metric_values, axis=0)
                std_values = sem(metric_values, axis=0)
                # Apply rolling mean with a window of 200
                avg_values = (
                    pd.Series(avg_values)
                    .rolling(window=200, min_periods=1)
                    .mean()
                    .values
                )
                std_values = (
                    pd.Series(std_values)
                    .rolling(window=200, min_periods=1)
                    .mean()
                    .values
                )
                # Plot the mean and std per timestep using the global time index X
                fig.add_trace(
                    go.Scatter(
                        x=X[: len(avg_values)],
                        y=avg_values,
                        mode="lines",
                        name=legend,
                        line=dict(width=2, color=col),
                        showlegend=(showlegend < 2),
                    ),
                    row=1,
                    col=list(metrics.keys()).index(metric_name) + 1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate(
                            [X[: len(avg_values)], X[: len(avg_values)][::-1]]
                        ),
                        y=np.concatenate(
                            [avg_values + std_values, (avg_values - std_values)[::-1]]
                        ),
                        fill="toself",
                        fillcolor=f"rgba({r},{g},{b},0.1)",
                        line=dict(width=0),
                        showlegend=False,
                    ),
                    row=1,
                    col=list(metrics.keys()).index(metric_name) + 1,
                )
                showlegend += 1
    fig.update_layout(
        title="Effect of added noise",
        xaxis_title="Time",
        yaxis_title="Value",
        template="plotly_white",
    )
    # Add y-axis title for the second column
    fig.update_yaxes(title_text="kW", row=1, col=2)
    # add time (x-axis) title for both subplots
    fig.update_xaxes(title_text="Time", row=1, col=2)
    fig.show()
    if save:
        fig.write_image(os.path.join(root, "noise_effect.png"))


def plot_setpoints_single_agent(
    base: str,
    configs: list[str] = ["just_solar", "just_battery"],
    metrics: dict = {
        "Solar Setpoints": "SolarAgent 0_setpoints",
        "Battery Setpoints": "BatteryAgent 0_setpoints",
    },
    scenario: str = "zero_noise",
    save: bool = False,
):
    """
    Plots the setpoints (mean) and (std) for Solar Setpoints in just_solar, and Battery Setpoints in just_battery.
    """
    root = "results"
    experiment_dir = os.path.join(root, "experiment")
    evaluate_dir = "evaluate"
    all_data = {}
    for config in configs:
        df = pd.read_csv(os.path.join(experiment_dir, base + config + ".csv"))
        uids = df["uid"].iloc[:-1].tolist()
        all_data[config] = {}
        for metric_name, metric_key in metrics.items():
            metric_data = []
            for uid in uids:
                metric_file = os.path.join(
                    evaluate_dir, scenario, metric_key, f"{uid}.npy"
                )
                if os.path.exists(metric_file):
                    data = np.load(metric_file)
                    metric_data.append(data)
                else:
                    logging.warning(f"Warning: {metric_file} does not exist.")
            if metric_data:
                all_data[config][metric_name] = np.array(metric_data)
            else:
                logging.warning(
                    f"Warning: No data found for metric {metric_name} in config {config}."
                )

    # plot the solar setpoints for just_solar (left) and battery setpoints for just_battery (right)
    fig = make_subplots(rows=1, cols=2, subplot_titles=list(metrics.keys()))
    solar = all_data["just_solar"]["Solar Setpoints"]
    battery = all_data["just_battery"]["Battery Setpoints"]
    avg_solar = np.mean(solar, axis=0)
    std_solar = sem(solar, axis=0)
    avg_battery = np.mean(battery, axis=0)
    std_battery = sem(battery, axis=0)

    # Apply rolling mean with a window of 100
    avg_solar = pd.Series(avg_solar).rolling(window=100, min_periods=1).mean().values
    std_solar = pd.Series(std_solar).rolling(window=100, min_periods=1).mean().values
    # set first avg_solar to the average because it is a massive outlier
    avg_solar[0] = np.mean(avg_solar[1:])
    avg_battery = (
        pd.Series(avg_battery).rolling(window=100, min_periods=1).mean().values
    )
    std_battery = (
        pd.Series(std_battery).rolling(window=100, min_periods=1).mean().values
    )
    # set first avg_battery to the average because it is a massive outlier
    avg_battery[0] = np.mean(avg_battery[1:])
    yellow = "#f0d003"  # hex color for yellow
    green = "#77905b"
    # Solar
    fig.add_trace(
        go.Scatter(
            x=X[: len(avg_solar)],
            y=avg_solar,
            mode="lines",
            name="Solar Setpoints",
            line=dict(width=2, color=yellow),  # yellow
        ),
        row=1,
        col=1,
    )
    r, g, b = hex_to_rgb(yellow)
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([X[: len(avg_solar)], X[: len(avg_solar)][::-1]]),
            y=np.concatenate([avg_solar + std_solar, (avg_solar - std_solar)[::-1]]),
            fill="toself",
            fillcolor=f"rgba({r},{g},{b},0.2)",  # yellow with transparency
            line=dict(width=0),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    # Battery
    fig.add_trace(
        go.Scatter(
            x=X[: len(avg_battery)],
            y=avg_battery,
            mode="lines",
            name="Battery Setpoints",
            line=dict(width=2, color=green),  # green
        ),
        row=1,
        col=2,
    )
    r, g, b = hex_to_rgb(green)
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([X[: len(avg_battery)], X[: len(avg_battery)][::-1]]),
            y=np.concatenate(
                [avg_battery + std_battery, (avg_battery - std_battery)[::-1]]
            ),
            fill="toself",
            fillcolor=f"rgba({r},{g},{b},0.2)",  # green with transparency
            line=dict(width=0),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    # Update layout
    fig.update_layout(
        title="Setpoints for Solar and Battery Agents",
        xaxis_title="Time",
        yaxis_title="Setpoint Value",
        template="plotly_white",
    )
    # Add x and y axis titles for both subplots
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_yaxes(title_text="Setpoint Value", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=1, col=2)
    fig.update_yaxes(title_text="Setpoint Value", row=1, col=2)
    fig.show()
    if save:
        fig.write_image(os.path.join(experiment_dir, "setpoints_single_agents.png"))


def print_metrics(
    base: str,
    configs: list[str] = [
        "3_handler",
        "baseline",
        "new_actor",
        "6_handler",
        "3_handler_default",
        "9_handler",
    ],
    scenario: str = "zero_noise",
    save: bool = False,
):
    """
    Prints the metrics for each config, including dispersion measures.
    """
    succesful_run = 1745
    metrics = {
        "Demand Satisfaction (%)": "grid_state",
        "Total CO2 emissions": "total_grid_production",
    }
    root = "results"
    experiment_dir = os.path.join(root, "experiment")
    evaluate_dir = "evaluate"

    # build columns: for each metric, add mean, var, std, CV, IQR, MAD
    cols = ["Config"]
    stats = ["mean", "var", "std", "cv", "iqr", "mad"]
    for m in metrics:
        for s in stats:
            cols.append(f"{m} ({s})")
    df = pd.DataFrame(columns=cols)

    # helper to compute all stats at once
    def compute_stats(arr: np.ndarray):
        arr = np.asarray(arr, dtype=float)
        mu = round(arr.mean(), 2)
        sigma2 = round(arr.var(ddof=0), 2)
        sigma = round(arr.std(ddof=0), 2)
        cv = round(sigma / mu, 2) if mu != 0 else np.nan
        return {
            "mean": mu,
            "var": sigma2,
            "std": sigma,
            "cv": cv,
            "iqr": round(iqr(arr), 2),
            "mad": round(median_abs_deviation(arr, scale="normal"), 2),
        }

    # load per-uid data
    for config in configs:
        # read list of UIDs
        cfg_csv = os.path.join(experiment_dir, base + config + ".csv")
        df_cfg = pd.read_csv(cfg_csv)
        uids = df_cfg["uid"].iloc[:-1].tolist()

        # gather per-uid metrics
        sat_vals = []
        prod_vals = []
        for uid in uids:
            # grid_state → satisfaction: % of timesteps where state<=0
            gs_path = (
                os.path.join(evaluate_dir, scenario, "grid_state", f"{uid}.npy")
                if config != "3_handler_default"
                else os.path.join(evaluate_dir, "default", "grid_state", f"{uid}.npy")
            )
            if os.path.exists(gs_path):
                state = np.load(gs_path)
                sat_vals.append(np.mean(state <= 0) * 100)
            else:
                logging.warning(f"Missing {gs_path}")

            # total_grid_production → CO2 (convert gCO2)
            gp_path = (
                os.path.join(
                    evaluate_dir, scenario, "total_grid_production", f"{uid}.npy"
                )
                if config != "3_handler_default"
                else os.path.join(
                    evaluate_dir, "default", "total_grid_production", f"{uid}.npy"
                )
            )
            if os.path.exists(gp_path):
                prod = np.load(gp_path)
                prod_vals.append(np.sum(prod) * 284.73)
            else:
                logging.warning(f"Missing {gp_path}")

        # compute all stats
        sat_stats = (
            compute_stats(np.array(sat_vals))
            if sat_vals
            else {s: np.nan for s in stats}
        )
        prod_stats = (
            compute_stats(np.array(prod_vals))
            if prod_vals
            else {s: np.nan for s in stats}
        )

        # build row
        row = {"Config": config}
        for s in stats:
            row[f"Demand Satisfaction (%) ({s})"] = sat_stats[s]
            row[f"Total CO2 emissions ({s})"] = prod_stats[s]

        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    # add the single "Successful Run" entry (no variance etc. — only mean=the value)
    gs = np.load(
        os.path.join(evaluate_dir, scenario, "grid_state", f"{succesful_run}.npy")
    )
    sat = np.mean(gs <= 0) * 100
    prod = (
        np.sum(
            np.load(
                os.path.join(
                    evaluate_dir,
                    scenario,
                    "total_grid_production",
                    f"{succesful_run}.npy",
                )
            )
        )
        * 284.73
    )

    row = {"Config": "Successful Run"}
    for s in stats:
        row[f"Demand Satisfaction (%) ({s})"] = sat if s == "mean" else np.nan
        row[f"Total CO2 emissions ({s})"] = prod if s == "mean" else np.nan
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    # print and optionally save
    print(df.to_string(index=False))
    if save:
        out_path = os.path.join(root, f"{base}_metrics.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved metrics to {out_path}")


def plot_behaviour(
    uid: int = 1745,
    days: int = 3,
    title: str = "Behaviour of a succesful agent",
    scenario: str = "zero_noise",
    save: bool = False,
):
    """
    1745 is a decent example of the the 'wanted' behaviour.
    Reading and plotting the evaluation data
    - setpoints for both solar and battery
    - battery intake and production
    - solar production
    - total demand
    - grid production
    """
    root = "evaluate"
    evaluate_dir = os.path.join(root, scenario)
    data = {
        "Solar Setpoints": "SolarAgent 0_setpoints",
        "Solar Production": "total_solar_production",
        "Battery Setpoints": "BatteryAgent 0_setpoints",
        "Battery Intake": "total_battery_intake",
        "Battery Production": "total_battery_production",
        "Total Demand": "total_demand",
        "Grid Production": "total_grid_production",
    }

    all_data = {}
    for metric_name, metric_key in data.items():
        metric_file = os.path.join(evaluate_dir, metric_key, f"{uid}.npy")
        if os.path.exists(metric_file):
            data = np.load(metric_file)
            all_data[metric_name] = data
        else:
            logging.warning(f"Warning: {metric_file} does not exist.")
    # Create a time index for the data
    time_index = pd.date_range(start="2025-09-01", periods=days * 24 * 60, freq="min")
    # Create one figure with all metrics
    fig = go.Figure()
    for metric_name, metric_data in all_data.items():
        # Apply rolling mean with a window of 100
        metric_data = (
            pd.Series(metric_data).rolling(window=100, min_periods=1).mean().values
        )
        fig.add_trace(
            go.Scatter(
                x=time_index[: len(metric_data)],
                y=metric_data,
                mode="lines",
                name=metric_name,
                line=dict(width=2),
            )
        )
    fig.update_layout(
        title=title, xaxis_title="Time", yaxis_title="kW", template="plotly_white"
    )
    fig.show()
    if save:
        fig.write_image(os.path.join(evaluate_dir, f"uid_{uid}_behaviour.png"))


def plot_performance(
    base: str,
    configs: list[str] = ["3_handler", "new_actor", "baseline"],
    metrics: dict = {
        "Reward": "reward",
        "CO2 emissions": "total_grid_production",
    },
    scenario: str = "zero_noise",
):
    """
    Plots 2 plots (side by side - 1 row 2 columns)
    Left has for both 3 handler and baseline the cumulative reward over time
    Right has for both 3 handler and baseline the cumulative grid production over time.
    Legend shows MADreamer (3-handler), MADreamer (SH), and PrioFlow.
    Also calculates mean ± SEM and plots shaded error regions.
    """
    root = "results"
    experiment_dir = os.path.join(root, "experiment")
    evaluate_dir = "evaluate"

    # load data
    all_data = {}
    for cfg in configs:
        df = pd.read_csv(os.path.join(experiment_dir, base + cfg + ".csv"))
        uids = df["uid"].iloc[:-1].tolist()
        all_data[cfg] = {}
        for name, key in metrics.items():
            runs = []
            for uid in uids:
                path = os.path.join(evaluate_dir, scenario, key, f"{uid}.npy")
                if os.path.exists(path):
                    runs.append(np.load(path))
                else:
                    logging.warning(f"Missing file: {path}")
            if runs:
                all_data[cfg][name] = np.stack(runs, axis=0)
            else:
                logging.warning(f"No data for {name} in {cfg}")

    # Prepare containers for cumulative stats
    cumulative_rewards = {}
    cumulative_rewards_sem = {}
    cumulative_co2 = {}
    cumulative_co2_sem = {}

    # First figure: instantaneous losses
    fig = make_subplots(rows=1, cols=2, subplot_titles=list(metrics.keys()))
    show_legend = True

    for metric_idx, (metric_name, _) in enumerate(metrics.items(), start=1):
        for cfg in configs:
            if metric_name not in all_data[cfg]:
                continue

            # raw runs: shape (n_runs, timesteps)
            runs = all_data[cfg][metric_name]

            # compute cumulative sums for later
            if metric_name == "Reward":
                cum = np.cumsum(runs, axis=1)
                cumulative_rewards[cfg] = cum.mean(axis=0)
                cumulative_rewards_sem[cfg] = sem(cum, axis=0)
            elif metric_name == "CO2 emissions":
                cum = np.cumsum(runs, axis=1)
                cumulative_co2[cfg] = cum.mean(axis=0)
                cumulative_co2_sem[cfg] = sem(cum, axis=0)

            # compute mean & sem for instantaneous values
            avg = runs.mean(axis=0)
            err = sem(runs, axis=0)

            # apply rolling smoothing
            avg = pd.Series(avg).rolling(window=100, min_periods=1).mean().values
            err = pd.Series(err).rolling(window=100, min_periods=1).mean().values

            steps = np.arange(len(avg))

            # pick color & legend
            if cfg == "3_handler":
                legend = "MADreamer"
                col = "#00e1d9"
            elif cfg == "baseline":
                legend = "PrioFlow"
                col = "#5e001f"
            else:
                legend = "MADreamer (SH)"
                col = "#DC758F"
            r, g, b = hex_to_rgb(col)

            # plot shaded error
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=avg + err,
                    fill=None,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=metric_idx,
            )
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=avg - err,
                    fill="tonexty",
                    mode="lines",
                    line=dict(width=0),
                    fillcolor=f"rgba({r},{g},{b},0.2)",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=metric_idx,
            )

            # plot mean line
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=avg,
                    mode="lines",
                    name=legend,
                    line=dict(color=col, width=2),
                    showlegend=show_legend,
                ),
                row=1,
                col=metric_idx,
            )

        show_legend = False

    fig.update_layout(title="Performance Over Time", template="plotly_white")
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=1, col=2)
    fig.update_yaxes(title_text="gCO2", row=1, col=2)

    fig.show()

    # Second figure: cumulative curves
    fig_cum = make_subplots(rows=1, cols=2, subplot_titles=list(metrics.keys()))
    show_legend = True

    for metric_idx, metric_name in enumerate(metrics.keys(), start=1):
        container = (
            (cumulative_rewards, cumulative_rewards_sem)
            if metric_name == "Reward"
            else (cumulative_co2, cumulative_co2_sem)
        )

        mean_dict, sem_dict = container

        for cfg in configs:
            if cfg not in mean_dict:
                continue

            mean_curve = mean_dict[cfg]
            err_curve = sem_dict[cfg]
            steps = np.arange(len(mean_curve))

            # color & legend
            if cfg == "3_handler":
                legend = "MADreamer"
                col = "#00e1d9"
            elif cfg == "baseline":
                legend = "PrioFlow"
                col = "#5e001f"
            else:
                legend = "MADreamer (SH)"
                col = "#DC758F"
            r, g, b = hex_to_rgb(col)

            # shaded region
            fig_cum.add_trace(
                go.Scatter(
                    x=steps,
                    y=(mean_curve + err_curve)
                    * (284.73 if metric_name == "CO2 emissions" else 1),
                    fill=None,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=metric_idx,
            )
            fig_cum.add_trace(
                go.Scatter(
                    x=steps,
                    y=(mean_curve - err_curve)
                    * (284.73 if metric_name == "CO2 emissions" else 1),
                    fill="tonexty",
                    mode="lines",
                    line=dict(width=0),
                    fillcolor=f"rgba({r},{g},{b},0.2)",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=metric_idx,
            )

            # mean line
            # convert CO2 to gCO2
            y = mean_curve * (284.73 if metric_name == "CO2 emissions" else 1)
            fig_cum.add_trace(
                go.Scatter(
                    x=steps,
                    y=y,
                    mode="lines",
                    name=legend,
                    line=dict(color=col, width=2),
                    showlegend=show_legend,
                ),
                row=1,
                col=metric_idx,
            )

        show_legend = False

    fig_cum.update_layout(title="Cumulative Performance", template="plotly_white")
    fig_cum.update_xaxes(title_text="Time", row=1, col=1)
    fig_cum.update_yaxes(title_text="Cumulative Reward", row=1, col=1)
    fig_cum.update_xaxes(title_text="Time", row=1, col=2)
    fig_cum.update_yaxes(title_text="Cumulative gCO2", row=1, col=2)

    fig_cum.show()


def plot_sankeys(
    base: str,
):
    """
    Sankey's are needed for validation for:
    - the single SimpleSetpoint (simple_setpoint)
    - baseline (prioflow)
    """
    # get the uids in from the experiment file
    root = "results"
    experiment_dir = os.path.join(root, "experiment")
    ss = pd.read_csv(os.path.join(experiment_dir, base + "simple_setpoint.csv"))
    prio = pd.read_csv(os.path.join(experiment_dir, base + "prioflow.csv"))
    uid_ss = ss["uid"].iloc[0]
    uid_prio = prio["uid"].iloc[0]
    # Using the Visualizer for SimpleSetpoint
    vis = Visualizer(uids=[uid_ss], selected_scenarios=["training"])
    vis.plot_metrics(metrics=["extensive_data"])
    # Using the Visualizer for Priorities
    vis = Visualizer(uids=[uid_prio], selected_scenarios=["training"])
    vis.plot_metrics(metrics=["extensive_data"])


def plot_actors(
    base: str,
    configs: list[str] = ["3_handler", "new_actor"],
    metrics: dict[str, str] = {
        "Reward": "reward",
        "CO2 emissions": "total_grid_production",
    },
    scenario: str = "zero_noise",
    save: bool = False,
):
    """
    Gather results from given actor configurations and plot the specified metrics
    comparing two actor implementations:
      - '3_handler'   -> fully shared
      - 'new_actor'   -> separate heads
    """
    root = "results"
    experiment_dir = os.path.join(root, "experiment")
    evaluate_dir = "evaluate"

    rewards = {}
    grid_production = {}
    for config in configs:
        df = pd.read_csv(os.path.join(experiment_dir, f"{base}{config}.csv"))
        uids = df["uid"].iloc[:-1].tolist()

        for metric_name, metric_key in metrics.items():
            metric_list = []
            for uid in uids:
                file_path = os.path.join(
                    evaluate_dir, scenario, metric_key, f"{uid}.npy"
                )
                if os.path.exists(file_path):
                    metric_list.append(np.load(file_path))
                else:
                    logging.warning(f"Missing file: {file_path}")
            if metric_list:
                # save reward and grid production data
                if metric_name == "Reward":
                    rewards[config] = np.array(metric_list)
                elif metric_name == "CO2 emissions":
                    grid_production[config] = np.array(metric_list)
            else:
                logging.warning(
                    f"No data for metric '{metric_name}' in config '{config}'"
                )

    # Setup subplots: 1 row, 2 columns
    fig = make_subplots(rows=1, cols=2, subplot_titles=list(metrics.keys()))

    show_legend = True
    # Plot reward
    for config, data in rewards.items():
        if config == "3_handler":
            legend = "MADreamer"
            col = "#00e1d9"
        if config == "new_actor":
            legend = "MADreamer (SH)"
            col = "#DC758F"
        r, g, b = hex_to_rgb(col)

        metric_values = np.mean(data, axis=0)
        std_values = sem(data, axis=0)
        # Apply rolling mean with a
        metric_values = (
            pd.Series(metric_values).rolling(window=100, min_periods=1).mean().values
        )
        # set first metric_values to mean because it is a massive outlier
        metric_values[0] = np.mean(metric_values[1:])
        std_values = (
            pd.Series(std_values).rolling(window=100, min_periods=1).mean().values
        )
        fig.add_trace(
            go.Scatter(
                x=X[: len(metric_values)],
                y=metric_values,
                mode="lines",
                name=legend,
                line=dict(width=2, color=col),
                showlegend=show_legend,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=np.concatenate(
                    [X[: len(metric_values)], X[: len(metric_values)][::-1]]
                ),
                y=np.concatenate(
                    [metric_values + std_values, (metric_values - std_values)[::-1]]
                ),
                fill="toself",
                fillcolor=f"rgba({r},{g},{b},0.1)",
                line=dict(width=0),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    show_legend = False  # Only show legend for the first trace
    # Plot grid production
    for config, data in grid_production.items():
        if config == "3_handler":
            legend = "MADreamer"
            col = "#00e1d9"
            r, g, b = hex_to_rgb(col)
        if config == "new_actor":
            legend = "MADreamer (SH)"
            col = "#DC758F"

        metric_values = np.mean(data, axis=0)
        std_values = sem(data, axis=0)
        # Apply rolling mean with a window of 50
        metric_values = (
            pd.Series(metric_values).rolling(window=100, min_periods=1).mean().values
        )
        # set first metric_values to mean because it is a massive outlier
        metric_values[0] = np.mean(metric_values[1:])
        std_values = (
            pd.Series(std_values).rolling(window=100, min_periods=1).mean().values
        )
        fig.add_trace(
            go.Scatter(
                x=X[: len(metric_values)],
                y=metric_values * 284.73,  # Convert to gCO2
                mode="lines",
                name=legend,
                line=dict(width=2, color=col),
                showlegend=show_legend,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=np.concatenate(
                    [X[: len(metric_values)], X[: len(metric_values)][::-1]]
                ),
                y=np.concatenate(
                    [
                        metric_values * 284.73 + std_values,
                        (metric_values * 284.73 - std_values)[::-1],
                    ]
                ),
                fill="toself",
                fillcolor=f"rgba({r},{g},{b},0.1)",
                line=dict(width=0),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    # Final layout adjustments
    fig.update_layout(
        title="Comparison of Actor Implementations", template="plotly_white"
    )
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Value")
    # If grid production needs units
    fig.update_yaxes(title_text="gCO2", row=1, col=2)

    fig.show()

    if save:
        output_path = os.path.join(root, "actor_comparison.png")
        fig.write_image(output_path)
        logging.info(f"Saved plot to {output_path}")


def do_statistical_test(base: str):
    """
    Using McNemar's test for Demand Satisfaction Ratio on a "demand_satisfied" boolean for each timestep
    Wilcoxon signed-rank test for CO2 emissions
    """
    # for both configs create a boolean series demand_satisfied where grid_state <= 0
    # for the wilcoxon test, we need the total CO2 emissions (grid_production) for each config
    root = "results"
    experiment_dir = os.path.join(root, "experiment")
    evaluate_dir = "evaluate"
    configs = ["3_handler", "new_actor", "baseline"]
    demand_satisfied = {}
    total_co2 = {}
    for config in configs:
        df = pd.read_csv(os.path.join(experiment_dir, base + config + ".csv"))
        uids = df["uid"].iloc[:-1].tolist()
        demand_satisfied[config] = []
        total_co2[config] = []
        for uid in uids:
            # load grid_state
            gs_path = os.path.join(
                evaluate_dir, "zero_noise", "grid_state", f"{uid}.npy"
            )
            if os.path.exists(gs_path):
                grid_state = np.load(gs_path)
                demand_satisfied[config].append(grid_state <= 0)
            else:
                logging.warning(f"Missing file: {gs_path}")

            # load total_grid_production
            gp_path = os.path.join(
                evaluate_dir, "zero_noise", "total_grid_production", f"{uid}.npy"
            )
            if os.path.exists(gp_path):
                total_co2[config].append(np.sum(np.load(gp_path)) * 284.73)
            else:
                logging.warning(f"Missing file: {gp_path}")
        # convert to numpy arrays
        demand_satisfied[config] = np.array(demand_satisfied[config])
        total_co2[config] = np.array(total_co2[config])
    # McNemar's test for demand satisfaction
    ds_3_handler = demand_satisfied["3_handler"].mean(axis=0)
    ds_new_actor = demand_satisfied["new_actor"].mean(axis=0)
    ds_baseline = demand_satisfied["baseline"].mean(axis=0)

    # for testing, put both in a pandas df and plot it as a line plot using plotly
    # ds_df = pd.DataFrame(
    #     {"3_handler": ds_3_handler, "new_actor": ds_new_actor, "baseline": ds_baseline}
    # )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(ds_3_handler)),
            y=ds_3_handler,
            mode="lines",
            name="3_handler",
            line=dict(color="#00e1d9", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(ds_new_actor)),
            y=ds_new_actor,
            mode="lines",
            name="new_actor",
            line=dict(color="#DC758F", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(ds_baseline)),
            y=ds_baseline,
            mode="lines",
            name="baseline",
            line=dict(color="#5e001f", width=2),
        )
    )
    fig.update_layout(
        title="Demand Satisfaction Ratio Comparison",
        xaxis_title="Time",
        yaxis_title="Demand Satisfaction Ratio",
        template="plotly_white",
    )
    fig.show()
    # Create a contingency table
    contingency_table = np.zeros((2, 2), dtype=int)
    contingency_table[0, 0] = np.sum(
        (ds_3_handler == 1) & (ds_new_actor == 1)
    )  # both satisfied
    contingency_table[0, 1] = np.sum(
        (ds_3_handler == 1) & (ds_new_actor == 0)
    )  # 3_handler satisfied, new_actor not
    contingency_table[1, 0] = np.sum(
        (ds_3_handler == 0) & (ds_new_actor == 1)
    )  # new_actor satisfied, 3_handler not
    contingency_table[1, 1] = np.sum(
        (ds_3_handler == 0) & (ds_new_actor == 0)
    )  # both not satisfied
    # Perform McNemar's test
    mcnemar_result = mcnemar(contingency_table, exact=True)
    print("McNemar's test result for Demand Satisfaction Ratio:\n")
    print(f"Statistic: {mcnemar_result.statistic}, p-value: {mcnemar_result.pvalue}")
    # Wilcoxon signed-rank test for CO2 emissions
    co2_3_handler = total_co2["3_handler"]
    co2_new_actor = total_co2["new_actor"]
    co2_baseline = total_co2["baseline"]
    # Ensure both arrays are of the same length
    if len(co2_3_handler) != len(co2_new_actor):
        logging.error(
            "CO2 emissions arrays are of different lengths, cannot perform Wilcoxon test."
        )
        return
    # Perform Wilcoxon signed-rank test
    wilcoxon_result = wilcoxon(co2_3_handler, co2_new_actor)
    print("\nWilcoxon signed-rank test result for CO2 emissions:\n")
    print(f"Statistic: {wilcoxon_result.statistic}, p-value: {wilcoxon_result.pvalue}")
    # also print medians
    median_3_handler = np.median(co2_3_handler)
    median_new_actor = np.median(co2_new_actor)
    median_baseline = np.median(co2_baseline)
    print(
        f"\nMedians:\n3_handler: {median_3_handler}, new_actor: {median_new_actor}, baseline: {median_baseline}"
    )

    # perform Wilcoxon signed-rank test for 3_handler and baseline
    wilcoxon_result = wilcoxon(co2_3_handler, co2_baseline)
    print(
        "\nWilcoxon signed-rank test result for CO2 emissions (3_handler vs baseline):\n"
    )
    print(f"Statistic: {wilcoxon_result.statistic}, p-value: {wilcoxon_result.pvalue}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot losses from multiple configs")
    parser.add_argument(
        "--base", type=str, required=True, help="Base name for the configs"
    )
    parser.add_argument(
        "--configs", nargs="+", required=False, help="List of config names to plot"
    )
    parser.add_argument(
        "--metrics", type=str, nargs="+", default=None, help="List of metrics to plot"
    )
    parser.add_argument("--save", action="store_true", help="Save the plots as images")

    args = parser.parse_args()

    metrics = None
    if args.metrics:
        metrics = {metric: metric.lower().replace(" ", "_") for metric in args.metrics}
    # print metrics
    print_metrics(base=args.base, save=args.save)
    print("Successfully printed metrics.")

    # plot losses
    plot_losses(base=args.base, metrics=metrics, save=args.save)
    print("Successfully plotted losses.")

    # plot noise effect
    plot_noise(base=args.base, save=args.save)
    print("Successfully plotted noise effect.")

    # plot setpoints for single agents
    plot_setpoints_single_agent(base=args.base, save=args.save)
    print("Successfully plotted setpoints for single agents.")

    # plot behaviour
    plot_behaviour(days=2)
    plot_behaviour(days=3)
    plot_behaviour(days=4)

    # plot behaviour prioflow
    plot_behaviour(uid=2663, days=2, title="Behaviour of PrioFlow agent")
    plot_behaviour(uid=2663, days=3, title="Behaviour of PrioFlow agent")
    plot_behaviour(uid=2663, days=4, title="Behaviour of PrioFlow agent")
    print("Successfully plotted behaviour.")

    # # plot performance
    plot_performance(base=args.base)
    print("Successfully plotted performance.")

    # plot sankeys
    plot_sankeys(base=args.base)
    print("Successfully plotted sankeys.")

    # plot actors
    plot_actors(base=args.base, save=args.save)
    print("Successfully plotted actors.")

    # do statistical test between 3_handler and new_actor
    do_statistical_test(base=args.base)
    print("Successfully performed statistical test between 3_handler and new_actor.")
