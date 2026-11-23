import argparse
import copy
import time

import numpy as np
import torch
from torch import nn
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.envs.utils import step_mdp
from torchrl.modules import MLP, QValueActor
from torchrl.objectives import DQNLoss, HardUpdate

from environment.custom_env import N_ACTIONS, OBS_SIZE
from training.common import (
    MODELS_DIR,
    REWARD_SCALE,
    TRAIN_EPISODES,
    TRANSITION_KEYS,
    VALIDATE_EVERY,
    final_evaluation,
    log_progress,
    make_env,
    prepare,
    run_many,
    save_run,
    set_seeds,
    train_seed,
    validate,
    write_summary,
)

BASE_CONFIG = {
    "learning_rate": 5e-4,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.995,
    "batch_size": 64,
    "buffer_size": 10_000,
    "target_update": 250,
    "learning_starts": 1_000,
    "hidden": [128, 128],
    "double_dqn": True,
    "grad_clip": 10.0,
    "seed": 1,
}
EXPERIMENTS = [
    ("dqn_01_baseline", {}),
    ("dqn_02_lr_1e-4", {"learning_rate": 1e-4}),
    ("dqn_03_lr_1e-3", {"learning_rate": 1e-3}),
    ("dqn_04_eps_0.99", {"epsilon_decay": 0.99}),
    ("dqn_05_eps_0.999", {"epsilon_decay": 0.999}),
    ("dqn_06_batch_32", {"batch_size": 32}),
    ("dqn_07_batch_128", {"batch_size": 128}),
    ("dqn_08_buffer_5k", {"buffer_size": 5_000}),
    ("dqn_09_buffer_20k", {"buffer_size": 20_000}),
    ("dqn_10_target_100", {"target_update": 100}),
    ("dqn_11_target_500", {"target_update": 500}),
    ("dqn_12_vanilla_dqn", {"double_dqn": False}),
]
SUMMARY_COLUMNS = ["learning_rate", "epsilon_decay", "batch_size", "buffer_size", "target_update", "double_dqn"]


def build_actor(hidden):
    return QValueActor(
        MLP(in_features=OBS_SIZE, out_features=N_ACTIONS, num_cells=hidden, activation_class=nn.ReLU),
        in_keys=["observation"],
        action_space="categorical",
        action_mask_key="action_mask",
    )


def greedy_action(actor):
    def select(td):
        return actor(td.clone())["action"].item()

    return select


def load_policy(path):
    checkpoint = torch.load(path, map_location="cpu")
    actor = build_actor(checkpoint["config"]["hidden"])
    actor.load_state_dict(checkpoint["state_dict"])
    actor.eval()
    return greedy_action(actor)


def train(experiment, episodes=TRAIN_EPISODES, verbose=True):
    name, overrides = experiment
    config = {**BASE_CONFIG, **overrides}
    torch.set_num_threads(1)
    set_seeds(config["seed"])
    rng = np.random.default_rng(config["seed"])
    env = make_env()
    actor = build_actor(config["hidden"])
    loss_module = DQNLoss(
        actor,
        action_space="categorical",
        delay_value=True,
        double_dqn=config["double_dqn"],
        loss_function="smooth_l1",
    )
    loss_module.make_value_estimator(gamma=config["gamma"])
    target_updater = HardUpdate(loss_module, value_network_update_interval=config["target_update"])
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=config["learning_rate"])
    buffer = TensorDictReplayBuffer(storage=LazyTensorStorage(config["buffer_size"]), batch_size=config["batch_size"])

    epsilon = config["epsilon_start"]
    history = {"episode_return": [], "episode_length": [], "loss": [], "q_value": [], "epsilon": [], "validation": []}
    best_return, best_state, total_steps = -np.inf, None, 0
    start = time.time()
    for episode in range(episodes):
        td = prepare(env.reset(seed=train_seed(config["seed"], episode)))
        ep_return, ep_len, losses, q_values = 0.0, 0, [], []
        while True:
            if rng.random() < epsilon:
                action = int(rng.choice(np.flatnonzero(td["action_mask"].numpy())))
            else:
                with torch.no_grad():
                    action = actor(td.clone())["action"].item()
            td["action"] = torch.tensor(action)
            td = prepare(env.step(td))
            ep_return += td["next", "reward"].item()
            transition = td.select(*TRANSITION_KEYS).clone()
            transition["next", "reward"] = transition["next", "reward"] * REWARD_SCALE
            buffer.add(transition)
            total_steps += 1
            ep_len += 1
            if total_steps >= config["learning_starts"]:
                batch = buffer.sample()
                loss = loss_module(batch)["loss"]
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), config["grad_clip"])
                optimizer.step()
                target_updater.step()
                losses.append(loss.item())
                with torch.no_grad():
                    q_values.append(actor(batch.select("observation", "action_mask").clone())["chosen_action_value"].mean().item())
            if td["next", "done"].item():
                break
            td = step_mdp(td)
        epsilon = max(config["epsilon_min"], epsilon * config["epsilon_decay"])
        history["episode_return"].append(ep_return)
        history["episode_length"].append(ep_len)
        history["loss"].append(float(np.mean(losses)) if losses else None)
        history["q_value"].append(float(np.mean(q_values)) if q_values else None)
        history["epsilon"].append(epsilon)
        if (episode + 1) % VALIDATE_EVERY == 0 or episode + 1 == episodes:
            score = validate(greedy_action(actor))
            history["validation"].append({"episode": episode + 1, "return": score, "steps": total_steps})
            if score > best_return:
                best_return, best_state = score, copy.deepcopy(actor.state_dict())
            if verbose:
                loss_text = f"{history['loss'][-1]:.4f}" if history["loss"][-1] is not None else "warmup"
                log_progress("DQN", name, episode, episodes, ep_return, f"eps {epsilon:.3f} loss {loss_text} val {score:8.1f}", start)

    actor.load_state_dict(best_state)
    train_seconds = time.time() - start
    MODELS_DIR.joinpath("dqn").mkdir(parents=True, exist_ok=True)
    torch.save({"config": config, "state_dict": best_state}, MODELS_DIR / "dqn" / f"{name}.pt")
    record = {
        "algorithm": "DQN",
        "name": name,
        "config": config,
        "episodes": episodes,
        "total_steps": total_steps,
        "train_seconds": train_seconds,
        "best_validation_return": best_return,
        "history": history,
        "evaluation": final_evaluation(greedy_action(actor)),
    }
    save_run("dqn", name, record)
    if verbose:
        std = record["evaluation"]["standard"]
        print(f"[DQN:{name}] done in {train_seconds:.0f}s | eval return {std['return_mean']:.1f} on-time {std['on_time_rate_mean']:.1%}", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description="Train DQN agents (TorchRL) on the maternal referral environment.")
    parser.add_argument("--experiments", nargs="*", default=None, help="experiment names to run (default: all)")
    parser.add_argument("--episodes", type=int, default=TRAIN_EPISODES)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    selected = [e for e in EXPERIMENTS if args.experiments is None or e[0] in args.experiments]
    records = run_many(_Runner(args.episodes), selected, args.workers)
    if args.experiments is None:
        write_summary("dqn", records, SUMMARY_COLUMNS)


class _Runner:
    def __init__(self, episodes):
        self.episodes = episodes

    def __call__(self, experiment):
        return train(experiment, self.episodes)


if __name__ == "__main__":
    main()
