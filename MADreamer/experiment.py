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

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import time
from scipy.stats import sem  # standard error of the mean


def plot_experiment_results(
    name: str,
    num_runs: int,
    experiment_dir: str = "results/experiment",
    results_root: str = "results",
    rolling_window: int = 100,
):
    """
    Loads experiment CSV to get UIDs, then for each metric loads all .npy runs,
    computes mean ± 1.96 * SEM for a 95% CI, applies optional rolling mean for
    selected metrics, and plots them in separate figures.

    Parameters:
    - name: substring to match CSV in experiment_dir
    - num_runs: expected number of .npy runs
    - experiment_dir: path to directory containing experiment CSVs
    - results_root: base path to metric .npy subdirectories
    - rolling_window: window size for rolling mean smoothing
    """
    # 1) CSV discovery
    matching = [f for f in os.listdir(experiment_dir) if name in f]
    if not matching:
        raise FileNotFoundError(f"No matching CSV in {experiment_dir} for '{name}'")
    df = pd.read_csv(os.path.join(experiment_dir, matching[0]))
    uids = df["uid"].iloc[:-1].tolist()
    if len(uids) != num_runs:
        print(f"Warning: found {len(uids)} UIDs but expected {num_runs}")

    # 2) Metrics definition
    metrics = {
        "Dynamics Loss": "dynamics_loss",
        "Prediction Loss": "prediction_loss",
        "Representation Loss": "representation_loss",
        "Critic Loss": "critic_loss",
        "Actor Loss": "actor_loss",
        # "Grid state":         "grid_state",
        # "Grid production":     "total_grid_production",
        # "Reward":              "reward",
    }

    # 3) Load runs, compute mean & 95% CI
    all_stats = {}
    for disp, sub in metrics.items():
        arrs = []
        for uid in uids:
            path = os.path.join(results_root, sub, f"{uid}.npy")
            arrs.append(np.load(path))
        arrs = np.stack(arrs, axis=0)  # (runs, timesteps)
        mu = arrs.mean(axis=0)
        se = sem(arrs, axis=0)  # shape (timesteps,)
        ci = 1.96 * se  # 95% CI half-width

        # Apply rolling mean smoothing for specific metrics
        if disp in ["Grid production", "Reward"] and rolling_window > 1:
            mu = (
                pd.Series(mu)
                .rolling(window=rolling_window, min_periods=1)
                .mean()
                .values
            )
            ci = (
                pd.Series(ci)
                .rolling(window=rolling_window, min_periods=1)
                .mean()
                .values
            )

        all_stats[disp] = {"mean": mu, "ci": ci, "steps": np.arange(mu.shape[0])}

    # 4) Plot each metric in a separate figure
    for disp_name, stats in all_stats.items():
        x = stats["steps"]
        mu = stats["mean"]
        ci = stats["ci"]

        fig = go.Figure()

        # mean line
        fig.add_trace(go.Scatter(x=x, y=mu, name=disp_name, mode="lines"))
        # 95% CI band
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([mu + ci, (mu - ci)[::-1]]),
                fill="toself",
                fillcolor="rgba(0,0,255,0.1)",
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
            )
        )
        fig.update_xaxes(title_text="Timestep")
        fig.update_yaxes(title_text=disp_name)
        fig.update_layout(
            title_text=f"{disp_name} — Experiment: {name} — 95% CI",
            # height=600, width=800
        )
        fig.show()


def train_one_algorithm(config: dict) -> tuple:
    """
    Trains a single algorithm and returns the uid of the algorithm and the training duration.
    """
    start_time = time.time()
    algorithm = BaseAlgorithm.get_algorithm(config["algorithm"], config)
    algorithm.train()
    Agent._id_counters = {}  # Reset agent ID counters after training
    end_time = time.time()
    training_duration = end_time - start_time
    return algorithm.saver.uid, training_duration


def eval_one_algorithm(uid: int, scenario_name: str) -> tuple:
    """
    Evaluates the algorithm linked to the provided uid and returns the cumulative reward and grid state.
    """
    config_path = f"results/configs/{uid}.yaml"
    if not os.path.exists(config_path):
        print(f"Configuration file not found for UID: {uid}")
        return 0.0, 0.0

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config_dir = f"evaluate/{scenario_name}/reward/{uid}.npy"
    if not os.path.exists(config_dir):
        print(f"Evaluating algorithm with UID: {uid}")
        config["eval_steps"] = config.get("eval_steps", 15000)
        start_time = datetime(2025, 9, 1, tzinfo=ZoneInfo("UTC"))
        config["start_time"] = start_time
        config["simulation_start_time"] = start_time
        config["extractor_type"] = "extensive"
        Agent._id_counters = {}
        algorithm = BaseAlgorithm.get_algorithm(
            config["algorithm"], config, evaluate=True
        )
        algorithm.eval()

    try:
        rewards = np.load(f"evaluate/{scenario_name}/reward/{uid}.npy", mmap_mode="r+")
        grid_state = np.load(
            f"evaluate/{scenario_name}/grid_state/{uid}.npy", mmap_mode="r+"
        )  # grid_state or total_grid_production
        grid_production = np.load(
            f"evaluate/{scenario_name}/total_grid_production/{uid}.npy", mmap_mode="r+"
        )
        # also get solar and battery production for percentages

    except FileNotFoundError:
        print(f"Evaluation files not found for UID: {uid}")
        return 0.0, 0.0, 0.0

    cumulative_reward = np.sum(rewards)
    cumulative_grid_state = np.sum(grid_state)
    cumulative_grid_production = np.sum(grid_production)
    return cumulative_reward, cumulative_grid_state, cumulative_grid_production


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot", action="store_true", help="Plot the results of the experiment"
    )
    parser.add_argument(
        "--name", type=str, required=True, help="Name of the experiment"
    )
    parser.add_argument(
        "--num", type=int, required=True, help="Number of runs for the experiment"
    )
    parser.add_argument(
        "--cfg",
        type=str,
        choices=[
            "just_battery",
            "just_solar",
            "dreamer",
            "3_handler",
            "january",
            "6_handler",
            "baseline",
            "new_actor",
            "ratio_1",
            "prioflow",
            "simple_setpoint",
            "9_handler",
        ],
        required=True,
        help="Configuration tag for the experiment",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="zero_noise",
        help="Scenario name for the experiment",
    )
    args = parser.parse_args()

    if args.plot:
        # Check all files in the experiment results directory for one that matches args.name
        experiment_dir = "results/experiment"
        matching_files = [f for f in os.listdir(experiment_dir) if args.name in f]

        if not matching_files:
            print(f"No matching experiment files found for name: {args.name}")
        else:
            # Assuming the first matching file is the one we want to plot
            plot_experiment_results(args.name, args.num, experiment_dir=experiment_dir)

    else:
        scenario_name = args.scenario
        exp_id = datetime.now().strftime("%Y%m%d%H")

        with open(f"data_scenarios/{scenario_name}.yaml", "r") as f:
            scenario = yaml.safe_load(f)

        with open(f"configs/{args.cfg}_config.yaml", "r") as f:
            base_config = yaml.safe_load(f)

        base_config["data_config"] = scenario
        base_config["eval_steps"] = 15000
        base_config["extractor_type"] = "extensive"
        base_config["device"] = "cpu"
        start_time = datetime(2025, 3, 1, tzinfo=ZoneInfo("UTC"))
        base_config["start_time"] = start_time
        base_config["simulation_start_time"] = start_time
        base_config["num_initial_random_steps"] = 0

        results = []
        failed = 0
        for run_idx in range(args.num):
            print(f"Running experiment {run_idx + 1}/{args.num}...")
            Agent._id_counters = {}

            # make a new config so base_config will not change through runs
            cfg = deepcopy(base_config)
            try:
                uid, training_time = train_one_algorithm(cfg)
            except Exception as e:
                print("Training failed:", e)
                failed += 1
                continue
            (
                cumulative_reward,
                cumulative_grid_state,
                cumulative_grid_production,
            ) = eval_one_algorithm(uid, scenario_name)

            results.append(
                {
                    "run_id": run_idx,
                    "uid": uid,
                    "cumulative_reward": cumulative_reward,
                    "cumulative_grid_state": cumulative_grid_state,
                    "cumulative_grid_production": cumulative_grid_production,
                    "training_time": training_time,
                }
            )
        if failed:
            print(f"\nFailed {failed} runs, retrying...\n")
            while failed > 0:
                Agent._id_counters = {}
                cfg = deepcopy(base_config)
                try:
                    uid, training_time = train_one_algorithm(cfg)
                    (
                        cumulative_reward,
                        cumulative_grid_state,
                        cumulative_grid_production,
                    ) = eval_one_algorithm(uid, scenario_name)

                    results.append(
                        {
                            "run_id": run_idx,
                            "uid": uid,
                            "cumulative_reward": cumulative_reward,
                            "cumulative_grid_state": cumulative_grid_state,
                            "cumulative_grid_production": cumulative_grid_production,
                            "training_time": training_time,
                        }
                    )
                    failed -= 1
                except Exception as e:
                    print("Training failed:", e)
                    continue

        # Calculate averages
        avg_reward = np.mean([result["cumulative_reward"] for result in results])
        avg_grid_state = np.mean(
            [result["cumulative_grid_state"] for result in results]
        )
        avg_grid_production = np.mean(
            [result["cumulative_grid_production"] for result in results]
        )
        avg_training_time = np.mean([result["training_time"] for result in results])

        # Save results to CSV
        os.makedirs("results/experiment", exist_ok=True)
        results_df = pd.DataFrame(results)
        results_df.loc[len(results_df)] = {
            "run_id": "average",
            "uid": exp_id,
            "cumulative_reward": avg_reward,
            "cumulative_grid_state": avg_grid_state,
            "cumulative_grid_production": avg_grid_production,
            "training_time": avg_training_time,
        }
        # hour and minute right now as a strin
        name = args.name + args.cfg
        results_df.to_csv(f"results/experiment/{name}.csv", index=False)

        print(f"Experiment completed. Results saved to results/experiment/{exp_id}.csv")
        # display the results
        print(results_df.head(len(results_df)))
