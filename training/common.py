import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.envs.libs.gym import GymWrapper

from environment.custom_env import MaternalReferralEnv
from training.evaluation import SCENARIOS, evaluate

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"
VALIDATION_SEEDS = list(range(5_000, 5_005))
TRAIN_EPISODES = 400
VALIDATE_EVERY = 25
REWARD_SCALE = 0.1
TRANSITION_KEYS = [
    "observation",
    "action_mask",
    "action",
    ("next", "observation"),
    ("next", "action_mask"),
    ("next", "reward"),
    ("next", "done"),
    ("next", "terminated"),
    ("next", "truncated"),
]


def make_env(scenario="standard"):
    return GymWrapper(MaternalReferralEnv(**SCENARIOS[scenario]), categorical_action_encoding=True)


def prepare(td):
    td["action_mask"] = td["action_mask"].bool()
    if ("next", "action_mask") in td.keys(include_nested=True):
        td["next", "action_mask"] = td["next", "action_mask"].bool()
    return td


def train_seed(run_seed, episode):
    return run_seed * 100_000 + episode


def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def obs_to_td(obs):
    return TensorDict(
        {
            "observation": torch.as_tensor(obs["observation"], dtype=torch.float32),
            "action_mask": torch.as_tensor(obs["action_mask"]).bool(),
        },
        [],
    )


def make_eval_policy(select_action):
    def act(env, obs):
        with torch.no_grad():
            return int(select_action(obs_to_td(obs)))

    return act


def validate(select_action):
    return evaluate(make_eval_policy(select_action), "standard", VALIDATION_SEEDS)["summary"]["return_mean"]


def final_evaluation(select_action):
    policy = make_eval_policy(select_action)
    return {scenario: evaluate(policy, scenario)["summary"] for scenario in SCENARIOS}


def save_run(algorithm, name, record):
    log_dir = LOGS_DIR / algorithm
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{name}.json").write_text(json.dumps(record, indent=2))


def log_progress(algorithm, name, episode, episodes, ep_return, extra, start):
    print(
        f"[{algorithm}:{name}] episode {episode + 1:>4}/{episodes} return {ep_return:8.1f} {extra} "
        f"({time.time() - start:5.0f}s)",
        flush=True,
    )


def run_many(train_fn, configs, workers):
    if workers <= 1:
        return [train_fn(c) for c in configs]
    with mp.get_context("spawn").Pool(workers) as pool:
        return pool.map(train_fn, configs)


def write_summary(algorithm, records, columns):
    lines = [",".join(["name", *columns, "val_best", "eval_return", "eval_on_time", "rainy_return", "rainy_on_time", "train_seconds"])]
    for r in records:
        std, rainy = r["evaluation"]["standard"], r["evaluation"]["rainy_season"]
        lines.append(
            ",".join(
                str(v)
                for v in [
                    r["name"],
                    *[r["config"][c] for c in columns],
                    round(r["best_validation_return"], 1),
                    round(std["return_mean"], 1),
                    round(std["on_time_rate_mean"], 4),
                    round(rainy["return_mean"], 1),
                    round(rainy["on_time_rate_mean"], 4),
                    round(r["train_seconds"], 1),
                ]
            )
        )
    path = LOGS_DIR / algorithm / "summary.csv"
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Saved {path}")
