from datetime import datetime
import logging
import os
import random
from zoneinfo import ZoneInfo
import torch
import yaml
import sys
import argparse
from filelock import FileLock
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from gui.eval import clear_all_files_with_uid
from marloes.agents.base import Agent
from src.marloes.algorithms.dreamer import Dreamer
from src.marloes.algorithms.base import BaseAlgorithm
from src.marloes.algorithms.priorities import Priorities
from src.marloes.algorithms.simplesetpoint import SimpleSetpoint
from src.marloes.algorithms.dyna import Dyna

search_space = {
    "horizon": lambda: random.choice([8, 16]),
    "batch_size": lambda: random.choice([32, 64, 128]),
    "replay_buffers.real_capacity": lambda: random.choice([100000]),
    "WorldModel.lr": lambda: random.choice([0.0001, 0.0002, 0.0005, 0.001, 0.005]),
    "WorldModel.weight_decay": lambda: random.choice([0.0, 0.001, 0.005]),
    "WorldModel.clip_grad": lambda: random.choice([1.0]),
    "WorldModel.free_bits": lambda: random.choice([0.5, 1.0]),
    "ActorCritic.actor_lr": lambda: random.choice(
        [0.0001, 0.0002, 0.0005, 0.001, 0.005]
    ),  # 0.01 throws errors (probably)
    "ActorCritic.critic_lr": lambda: random.choice(
        [0.0001, 0.0002, 0.0005, 0.001, 0.005]
    ),
    "ActorCritic.actor_weight_decay": lambda: random.choice([0.0, 0.001, 0.005]),
    "ActorCritic.critic_weight_decay": lambda: random.choice([0.0, 0.001, 0.005]),
    "ActorCritic.actor_clip_grad": lambda: random.choice([1]),
    "ActorCritic.critic_clip_grad": lambda: random.choice([1]),
    "ActorCritic.gamma": lambda: random.choice([0.997, 0.999]),
    "ActorCritic.lambda": lambda: random.choice([0.98, 0.99]),
    "ActorCritic.entropy_coef": lambda: random.choice([0.1, 0.01]),
}


def sample_config(base_config, overrides):
    cfg = yaml.safe_load(yaml.safe_dump(base_config))  # deep copy
    for path, val in overrides.items():
        d = cfg
        *head, last = path.split(".")
        for h in head:
            d = d.setdefault(h, {})
        d[last] = val
    return cfg


def random_search(base_config, n_trials=20):
    configs = []
    for _ in range(n_trials):
        samp = {k: fn() for k, fn in search_space.items()}
        configs.append(sample_config(base_config, samp))
    return configs


def run_single_trial(config):
    """
    Single trial run for a given configuration, always starting from the first of March 2025.
    """
    config["device"] = "cpu"
    start_time = datetime(2025, 3, 1, tzinfo=ZoneInfo("UTC"))
    config["start_time"] = start_time
    algorithm = BaseAlgorithm.get_algorithm(config["algorithm"], config)
    algorithm.train()
    Agent._id_counters = {}
    return algorithm.saver.uid


def decide_worker_count(user_workers):
    if user_workers is not None:
        return max(1, user_workers)
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return min(4, mp.cpu_count())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--analyze", nargs="?", const=True)
    parser.add_argument(
        "--cfg",
        type=str,
        default="dreamer",
        choices=["just_battery", "just_solar", "dreamer", "new_actor"],
        help="Path to the base configuration file",
    )
    parser.add_argument("--no-parallel", dest="parallel", action="store_false")
    parser.add_argument("--workers", type=int, help="Number of worker processes")
    parser.add_argument(
        "--num", type=int, default=10, help="Number of trials to run in search mode"
    )
    args = parser.parse_args()

    scenario_name = "zero_noise"

    with open(f"data_scenarios/{scenario_name}.yaml", "r") as f:
        scenario = yaml.safe_load(f)

    if args.search:
        with open(f"configs/{args.cfg}_config.yaml") as f:
            base = yaml.safe_load(f)
        base["data_config"] = scenario

        trials_cfgs = random_search(base, n_trials=args.num)

        with FileLock("results/hyperparam_uid.txt.lock"):
            with open("results/hyperparam_uid.txt") as f:
                hyperparam_uid = int(f.read().strip())
            with open("results/hyperparam_uid.txt", "w") as f:
                f.write(str(hyperparam_uid + 1))

        os.makedirs("results/hyperparams", exist_ok=True)
        f_out = open(f"results/hyperparams/{hyperparam_uid}.txt", "a")

        if args.parallel:
            n_workers = decide_worker_count(args.workers)
            mp.set_start_method("spawn", force=True)
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = [pool.submit(run_single_trial, cfg) for cfg in trials_cfgs]
                for fut in futures:
                    run_uid = fut.result()
                    print(f"Finished trial → UID {run_uid}")
                    f_out.write(f"{run_uid}\n")
                    f_out.flush()
        else:
            i = 0
            for cfg in trials_cfgs:
                i += 1
                failed_dir = "failed_configs"
                try:
                    run_uid = run_single_trial(cfg)
                    print(f"Finished trial → UID {run_uid}")
                    f_out.write(f"{run_uid}\n")
                    f_out.flush()
                except Exception as e:
                    os.makedirs(failed_dir, exist_ok=True)
                    with open(f"{failed_dir}/{i}.yaml", "w") as failed_file:
                        yaml.safe_dump(cfg, failed_file)
                    print(
                        f"Trial {i} failed: {e}. Saved config to {failed_dir}/{i}.yaml"
                    )

        f_out.close()

    if args.analyze:
        if args.analyze is True:
            with open("results/hyperparam_uid.txt", "r") as f:
                hyperparam_uid = int(f.read().strip()) - 1
        else:
            hyperparam_uid = args.analyze

        print(f"Analyzing hyperparam uid: {hyperparam_uid}")
        with open(f"results/hyperparams/{hyperparam_uid}.txt", "r") as f:
            run_uids = [line.strip() for line in f.readlines()]

        rewards = {}
        grid_production = {}
        for i, run_uid in enumerate(run_uids):
            with open(f"results/configs/{run_uid}.yaml", "r") as f:
                config = yaml.safe_load(f)

            config_dir = f"evaluate/{scenario_name}/reward/{run_uid}.npy"
            if not os.path.exists(config_dir):
                print(f"Evaluating run uid: {run_uid}, {i + 1} of {len(run_uids)}")
                config["data_config"] = scenario
                config["uid"] = int(run_uid)
                config["eval_steps"] = 20000
                start_time = datetime(2025, 9, 1, tzinfo=ZoneInfo("UTC"))
                config["start_time"] = start_time
                config["simulation_start_time"] = start_time
                config["extractor_type"] = "extensive"
                Agent._id_counters = {}
                logging.getLogger().setLevel(logging.WARNING)
                algorithm = BaseAlgorithm.get_algorithm(
                    config["algorithm"], config, evaluate=True
                )
                algorithm.eval()
                logging.getLogger().setLevel(logging.INFO)

            try:
                rewards[run_uid] = np.load(
                    f"evaluate/{scenario_name}/reward/{run_uid}.npy", mmap_mode="r+"
                )
                grid_production[run_uid] = np.load(
                    f"evaluate/{scenario_name}/total_grid_production/{run_uid}.npy",
                    mmap_mode="r+",
                )
            except FileNotFoundError:
                print(f"File not found for run uid: {run_uid}")
                continue

            rewards[run_uid] = np.sum(rewards[run_uid])
            grid_production[run_uid] = np.sum(grid_production[run_uid])

        sorted_uids = sorted(
            rewards.items(), key=lambda x: x[1], reverse=False
        )  # rewards are negative
        sorted_grid_production = sorted(
            grid_production.items(), key=lambda x: x[1], reverse=True
        )

        print("Sorted UIDs based on summed rewards")
        for uid, reward in sorted_uids:
            print(f"UID: {uid}, Reward over the last two days: {reward}")

        print("Sorted UIDs based on summed grid production")
        for uid, grid_prod in sorted_grid_production:
            print(f"UID: {uid}, Grid production over the last two days: {grid_prod}")
