import argparse
from datetime import datetime, timedelta
import logging
from zoneinfo import ZoneInfo
import yaml
import os
import copy
import random
import numpy as np
import torch

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from src.marloes.algorithms.dreamer import Dreamer
from src.marloes.algorithms.base import BaseAlgorithm
from src.marloes.algorithms.prioflow import PrioFlow
from src.marloes.algorithms.simplesetpoint import SimpleSetpoint
from src.marloes.algorithms.dyna import Dyna
from marloes.handlers.base import Handler

search_space = {
    "dyna.updates_per_step": lambda: random.choice([15, 18, 20]),
    "SAC.weight_decay": lambda: 10 ** np.random.uniform(-8, -5),
    "SAC.tau": lambda: random.choice([0.005, 0.01, 0.02, 0.025]),
    "dyna.key": lambda: random.choice([4, 6, 8, 10]),
    "SAC.actor_lr": lambda: 10 ** np.random.uniform(-6, -4),
    "SAC.critic_lr": lambda: 10 ** np.random.uniform(-6, -4),
    "SAC.value_lr": lambda: 10 ** np.random.uniform(-6, -4),
    "SAC.alpha_lr": lambda: 10 ** np.random.uniform(-6, -4),
    "SAC.eps": lambda: 10 ** np.random.uniform(-8, -6),
    "batch_size": lambda: random.choice([32, 64, 128]),
    "SAC.critic_actor_update_ratio": lambda: random.choice([1, 2, 3]),
    "SAC.gamma": lambda: random.choice([0.98, 0.99, 0.995, 0.999]),
    "SAC.hidden_dim": lambda: random.choice([64, 128, 256]),
    "SAC.num_layers": lambda: random.choice([1, 2, 3]),
    "WorldModel.lr": lambda: 10 ** np.random.uniform(-4, -2),
    "WorldModel.world_hidden_size": lambda: random.choice([32, 64, 128]),
    "WorldModel.forecast_hidden_size": lambda: random.choice([8, 16, 32]),
}


def run_training(config, scenario_name):
    """Run a single trial with the given configuration."""
    start_time = datetime(2025, 1, 1, tzinfo=ZoneInfo("UTC"))
    random_minutes = random.randint(0, 4 * 30 * 24 * 60)
    start_time += timedelta(minutes=random_minutes)
    config["start_time"] = start_time
    config["simulation_start_time"] = start_time
    config["data_config"] = get_data_scenario(scenario_name)
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    algorithm = BaseAlgorithm.get_algorithm(config["algorithm"], config)
    algorithm.train()
    Handler._id_counters = {}
    print(
        f"Training completed for {scenario_name} with UID: {algorithm.saver.uid}",
        flush=True,
    )
    return algorithm.saver.uid


def run_evaluation(config, scenario_name, uid):
    """Run a single evaluation with the given configuration."""
    config["data_config"] = get_data_scenario(scenario_name)
    config["uid"] = uid
    config["eval_steps"] = 50000
    start_time = datetime(2025, 9, 1, tzinfo=ZoneInfo("UTC"))
    config["start_time"] = start_time
    config["simulation_start_time"] = start_time
    config["extractor_type"] = "extensive"  # TODO: Fix this error
    Handler._id_counters = {}
    logging.getLogger().setLevel(logging.WARNING)
    algorithm = BaseAlgorithm.get_algorithm(config["algorithm"], config, evaluate=True)
    algorithm.eval()
    logging.getLogger().setLevel(logging.INFO)
    print(
        f"Evaluation completed for {scenario_name} with UID: {algorithm.saver.uid}",
        flush=True,
    )
    return algorithm.saver.uid


def run_single_trial(config, scenario_name):
    eval_config = copy.deepcopy(config)
    uid = run_training(config, scenario_name)
    run_evaluation(eval_config, scenario_name, uid)
    return uid


def load_base_config(config_file):
    with open(config_file) as f:
        return yaml.safe_load(f)


def get_data_scenario(scenario_name):
    with open(f"data_scenarios/{scenario_name}.yaml", "r") as f:
        scenario = yaml.safe_load(f)
    return scenario


def update_config(config, args, **kwargs):
    """Helper function to update the config dictionary with parameters."""
    for key, value in kwargs.items():
        if value is not None:
            if "." in key:
                default = config
                *parts, last = key.split(".")
                for part in parts:
                    default = default.setdefault(part, {})
                default[last] = value
            else:
                config[key] = value


def decide_worker_count(user_workers):
    if user_workers is not None:
        return max(1, user_workers)
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return min(4, mp.cpu_count())


def parallel_trials(callable_fn, list_of_args, workers):
    """Runs callables in parallel and returns results in order of input."""
    results = [None] * len(list_of_args)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(callable_fn, *args): i for i, args in enumerate(list_of_args)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                print(f"Trial {idx} failed: {e}", flush=True)
    return results


def serial_trials(callable_fn, list_of_args):
    return [callable_fn(*args) for args in list_of_args]


def hyperparam_search(args):
    base = load_base_config(args.config)
    jobs = []
    for i in range(args.trials):
        overrides = {key: value() for key, value in search_space.items()}
        config = yaml.safe_load(yaml.safe_dump(base))
        for path, val in overrides.items():
            update_config(config, args, **{path: val})
        config["seed"] = args.seed_start + i
        jobs.append((config, "default"))

    workers = decide_worker_count(args.workers) if args.parallel else 1
    trial_fn = run_single_trial
    runner = parallel_trials if args.parallel and workers > 1 else serial_trials
    results = (
        runner(trial_fn, jobs, workers)
        if args.parallel and workers > 1
        else runner(trial_fn, jobs)
    )

    for i, uid in enumerate(results):
        print(f"HP_SEARCH | Trial {i} | UID: {uid}", flush=True)


def ablation(args):
    base = load_base_config(args.config)
    jobs = []
    for use_gru in [True, False]:
        for seed in range(args.seed_start, args.seed_end + 1):
            config = yaml.safe_load(yaml.safe_dump(base))
            config["seed"] = seed
            update_config(config, args, **{"WorldModel.use_gru": use_gru})
            jobs.append((config, "default"))

    workers = decide_worker_count(args.workers) if args.parallel else 1
    trial_fn = run_single_trial
    runner = parallel_trials if args.parallel and workers > 1 else serial_trials
    results = (
        runner(trial_fn, jobs, workers)
        if args.parallel and workers > 1
        else runner(trial_fn, jobs)
    )

    for i, uid in enumerate(results):
        label = "withGRU" if i < ((args.seed_end - args.seed_start + 1)) else "noGRU"
        print(f"ABLATION_{label} | Trial {i} | UID: {uid}", flush=True)


def paradigm(args):
    base = load_base_config(args.config)
    base_6 = load_base_config("configs/dyna_config_6.yaml")
    base_12 = load_base_config("configs/dyna_config_12.yaml")
    jobs = []
    for sctde in [False, True]:
        for n_assets in [3, 6, 12]:
            for seed in range(args.seed_start, args.seed_end + 1):
                if n_assets == 3:
                    config = yaml.safe_load(yaml.safe_dump(base))
                elif n_assets == 6:
                    config = yaml.safe_load(yaml.safe_dump(base_6))
                elif n_assets == 12:
                    config = yaml.safe_load(yaml.safe_dump(base_12))
                config["seed"] = seed
                update_config(config, args, **{"dyna.sCTDE": sctde})
                jobs.append((config, "default"))

    workers = decide_worker_count(args.workers) if args.parallel else 1
    trial_fn = run_single_trial
    runner = parallel_trials if args.parallel and workers > 1 else serial_trials
    results = (
        runner(trial_fn, jobs, workers)
        if args.parallel and workers > 1
        else runner(trial_fn, jobs)
    )

    for i, uid in enumerate(results):
        print(f"PARADIGM | Trial {i} | UID: {uid}", flush=True)


def main_exp(args):
    base = load_base_config(args.config)
    configs = [
        {"algorithm": "PrioFlow"},
        {
            "dyna.sCTDE": True,
            "dyna.real_sample_ratio": 1.0,
            "dyna.k": 0,
            "dyna.world_model_update_frequency": 50000,
        },
        {"dyna.sCTDE": True, "dyna.real_sample_ratio": 0.8},
        {"dyna.sCTDE": True, "dyna.real_sample_ratio": 0.5},
        {
            "dyna.sCTDE": False,
            "dyna.real_sample_ratio": 1.0,
            "dyna.k": 0,
            "dyna.world_model_update_frequency": 50000,
        },
        {"dyna.sCTDE": False, "dyna.real_sample_ratio": 0.8},
        {"dyna.sCTDE": False, "dyna.real_sample_ratio": 0.5},
    ]
    jobs = []
    for data_scenario in ["zero_noise", "default"]:
        for conf in configs:
            for seed in range(args.seed_start, args.seed_end + 1):
                config = yaml.safe_load(yaml.safe_dump(base))
                config["seed"] = seed
                update_config(config, args, **conf)
                jobs.append((config, data_scenario))

    workers = decide_worker_count(args.workers) if args.parallel else 1
    trial_fn = run_single_trial
    runner = parallel_trials if args.parallel and workers > 1 else serial_trials
    results = (
        runner(trial_fn, jobs, workers)
        if args.parallel and workers > 1
        else runner(trial_fn, jobs)
    )

    for i, uid in enumerate(results):
        print(
            f"MAIN_EXP | Trial {i} | UID: {uid} | Config: {jobs[i][0]} | Scenario: {jobs[i][1]}",
            flush=True,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        required=True,
        choices=["hyperparam_search", "ablation", "paradigm", "main"],
    )
    parser.add_argument("--config", type=str, default="configs/dyna_config.yaml")
    parser.add_argument("--seed_start", type=int, default=10)
    parser.add_argument("--seed_end", type=int, default=19)
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of trials for hyperparameter search",
    )
    parser.add_argument(
        "--parallel", action="store_true", help="Enable multiprocessing"
    )
    parser.add_argument("--workers", type=int, help="Number of parallel workers")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.experiment == "hyperparam_search":
        hyperparam_search(args)
    elif args.experiment == "ablation":
        ablation(args)
    elif args.experiment == "paradigm":
        paradigm(args)
    elif args.experiment == "main":
        main_exp(args)


if __name__ == "__main__":
    main()
