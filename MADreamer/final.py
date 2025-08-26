"""
This is a script to run over the weekend.
Runs multiple experiments, with different configurations after each other, running a series of subprocesses.
- 1: python experiment.py --name <name> --cfg just_solar --num <num>
- 2: python experiment.py --name <name> --cfg just_battery --num <num>
- 3: python experiment.py --name <name> --cfg experiment --num <num>
- 4: python experiment.py --name <name> --cfg experiment --scenario default --num <num>
- 5: python experiment.py --name <name> --cfg baseline --num <num>
- 6: python experiment.py --name <name> --cfg 6_agents --num <num>
All configurations must be checked if they are correct
"""

import os
import subprocess
import argparse
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

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import time
from scipy.stats import sem  # standard error of the mean

NAME = "final_"


def run_experiment(name, cfg_name, num_trials):
    """
    Run a single experiment with the specified configuration name and number of trials.
    """
    cmd = f"python experiment.py --name {name} --cfg {cfg_name} --num {num_trials}"
    print(f"Running command: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def plot_experiment(name, cfg_name, num_trials):
    """
    Plot the results of a single experiment using the experiment.py plotting flag.
    """
    # Construct the plotting command; cfg_name may include a scenario flag
    cmd = (
        f"python experiment.py --plot --name {name} --cfg {cfg_name} --num {num_trials}"
    )
    print(f"Plotting results with: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def plot_all_results(
    experiments: dict,
    experiment_dir: str = "results/experiment",
    results_root: str = "results",
    rolling_window: int = 100,
):
    """
    Aggregate and plot cumulative reward and cumulative grid production
    for all configurations as two separate figures, including 95% CIs.
    """
    all_stats = {}
    # experiments.update(
    #     {
    #         "add_": 10,
    #         "add_1": 5,
    #         "add_2": 5,
    #         "add_3": 10,
    #         "add_4": 5,
    #         "add_5": 5,
    #         "add_6": 5,
    #         "add_7": 10,
    #         "add_8": 10,
    #         "add_9": 10,
    #         "old_": 10,
    #         "old_1": 5,
    #         "old_2": 10,
    #         "old_3": 10,
    #         "old_4": 10,
    #     }
    # )
    for cfg_name, num_runs in experiments.items():
        # Determine the run name used in filenames
        if cfg_name == "3_handler --scenario default":
            run_name = NAME + "3_handler_default"
        elif "add_" in cfg_name:
            # extract whwatever is after add_ as str
            num = cfg_name.split("add_")[1] if len(cfg_name.split("add_")) > 1 else ""
            run_name = "NewActor" + num + "new_actor"
        elif "old_" in cfg_name:
            # extract whatever is after old_ as str
            num = cfg_name.split("old_")[1] if len(cfg_name.split("old_")) > 1 else ""
            run_name = "OldActor" + num + "dreamer"
        else:
            run_name = NAME + cfg_name

        # Locate the matching CSV to extract UIDs
        matching = [
            f
            for f in os.listdir(experiment_dir)
            if f.startswith(run_name) and f.endswith(".csv")
        ]
        if not matching:
            print(
                f"Warning: no CSV found for '{run_name}' in {experiment_dir}, skipping {cfg_name}"
            )
            continue
        df = pd.read_csv(os.path.join(experiment_dir, matching[0]))
        uids = df["uid"].iloc[:-1].tolist()
        if len(uids) != num_runs:
            print(
                f"Warning: found {len(uids)} UIDs for '{cfg_name}' but expected {num_runs}"
            )

        # Load and accumulate metric arrays
        reward_runs = [
            np.load(os.path.join(results_root, "reward", f"{uid}.npy")) for uid in uids
        ]
        grid_runs = [
            np.load(os.path.join(results_root, "total_grid_production", f"{uid}.npy"))
            for uid in uids
        ]

        reward_cum = np.cumsum(np.stack(reward_runs, axis=0), axis=1)
        grid_cum = np.cumsum(np.stack(grid_runs, axis=0), axis=1)

        # Compute mean and 95% CI half-width
        mu_reward = reward_cum.mean(axis=0)
        ci_reward = 1.96 * sem(reward_cum, axis=0)
        mu_grid = grid_cum.mean(axis=0)
        ci_grid = 1.96 * sem(grid_cum, axis=0)

        # Optional rolling mean smoothing
        if rolling_window > 1:
            mu_reward = (
                pd.Series(mu_reward)
                .rolling(window=rolling_window, min_periods=1)
                .mean()
                .values
            )
            ci_reward = (
                pd.Series(ci_reward)
                .rolling(window=rolling_window, min_periods=1)
                .mean()
                .values
            )
            mu_grid = (
                pd.Series(mu_grid)
                .rolling(window=rolling_window, min_periods=1)
                .mean()
                .values
            )
            ci_grid = (
                pd.Series(ci_grid)
                .rolling(window=rolling_window, min_periods=1)
                .mean()
                .values
            )

        all_stats[cfg_name] = {
            "steps": np.arange(mu_reward.shape[0]),
            "mu_reward": mu_reward,
            "ci_reward": ci_reward,
            "mu_grid": mu_grid,
            "ci_grid": ci_grid,
        }

    # Plot cumulative reward figure with CI band
    fig_reward = go.Figure()
    for cfg, stats in all_stats.items():
        x = stats["steps"]
        mu = stats["mu_reward"]
        ci = stats["ci_reward"]
        # Mean line
        fig_reward.add_trace(go.Scatter(x=x, y=mu, name=f"{cfg}", mode="lines"))
        # CI shading
        fig_reward.add_trace(
            go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([mu + ci, (mu - ci)[::-1]]),
                fill="toself",
                fillcolor="rgba(0,128,255,0.1)",  # More blueish color
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
            )
        )
    fig_reward.update_xaxes(title_text="Timestep")
    fig_reward.update_yaxes(title_text="Cumulative Reward")
    fig_reward.update_layout(
        title_text="Cumulative Reward Across Configurations",
        # height=600, width=1000
    )
    fig_reward.show()

    # Plot cumulative grid production figure with CI band
    fig_grid = go.Figure()
    for cfg, stats in all_stats.items():
        x = stats["steps"]
        mu = stats["mu_grid"]
        ci = stats["ci_grid"]
        # Mean line
        fig_grid.add_trace(go.Scatter(x=x, y=mu, name=f"{cfg}", mode="lines"))
        # CI shading
        fig_grid.add_trace(
            go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([mu + ci, (mu - ci)[::-1]]),
                fill="toself",
                fillcolor="rgba(0,128,255,0.1)",  # More blueish color
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
            )
        )
    fig_grid.update_xaxes(title_text="Timestep")
    fig_grid.update_yaxes(title_text="Cumulative Grid Production")
    fig_grid.update_layout(
        title_text="Cumulative Grid Production Across Configurations",
        # height=600, width=1000
    )
    fig_grid.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run experiments with different configurations."
    )
    parser.add_argument(
        "--num", type=int, default=1, help="Number of trials for each experiment."
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Flag to plot results after running experiments.",
    )
    args = parser.parse_args()

    NUM = args.num

    experiments = {
        # "just_solar": NUM,
        # "just_battery": NUM,
        "3_handler": NUM,
        # "3_handler --scenario default": NUM,
        # "baseline": NUM,
        # "6_handler": NUM,
        # "simple_setpoint": 1,
        "new_actor": NUM,
        # "prioflow": 1,
        # '9_handler': NUM,
    }
    if args.plot:
        # Plot all configurations in one combined figure
        plot_all_results(experiments)
    else:
        # 1) Run experiments for each configurations
        for cfg_name, num_trials in experiments.items():
            try:
                if cfg_name == "3_handler --scenario default":
                    run_experiment(NAME + "3_handler_default", cfg_name, num_trials)
                else:
                    run_experiment(NAME, cfg_name, num_trials)
            except subprocess.CalledProcessError as e:
                print(f"An error occurred while running '{cfg_name}': {e}")

        print("All experiments attempted.")

        # 2) Plot results for each configuration
        for cfg_name in experiments.keys():
            try:
                if cfg_name == "3_handler --scenario default":
                    plot_experiment(NAME + "3_handler_default", cfg_name, NUM)
                else:
                    plot_experiment(NAME + cfg_name, cfg_name, NUM)
            except subprocess.CalledProcessError as e:
                print(f"An error occurred while plotting '{cfg_name}': {e}")

    print("Plotting complete.")

    # subprocess plot.py
    cmd = f"python plot.py --save --base {NAME}"
    print(f"Running command: {cmd}")
    subprocess.run(cmd, shell=True, check=True)
