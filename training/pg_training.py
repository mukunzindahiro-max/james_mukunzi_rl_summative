import argparse
import copy
import functools
import json
import time

import numpy as np
import torch
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.envs.utils import step_mdp
from torchrl.modules import MLP, MaskedCategorical, ProbabilisticActor, ValueOperator
from torchrl.objectives import A2CLoss, ClipPPOLoss
from torchrl.objectives.value import GAE

from environment.custom_env import N_ACTIONS, OBS_SIZE
from training.common import (
    LOGS_DIR,
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

BASE_CONFIGS = {
    "reinforce": {
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "entropy_coef": 0.01,
        "hidden": [128, 128],
        "normalize_returns": True,
        "value_baseline": False,
        "episodes_per_update": 1,
        "grad_clip": 1.0,
        "seed": 1,
    },
    "ppo": {
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "entropy_coef": 0.01,
        "value_coef": 0.5,
        "rollout_steps": 1024,
        "epochs": 4,
        "minibatch_size": 128,
        "hidden": [128, 128],
        "grad_clip": 0.5,
        "seed": 1,
    },
    "a2c": {
        "learning_rate": 7e-4,
        "gamma": 0.99,
        "gae_lambda": 1.0,
        "entropy_coef": 0.01,
        "value_coef": 0.5,
        "rollout_steps": 16,
        "normalize_advantage": True,
        "hidden": [128, 128],
        "grad_clip": 0.5,
        "seed": 1,
    },
}
EXPERIMENTS = {
    "reinforce": [
        ("reinforce_01_baseline", {}),
        ("reinforce_02_lr_3e-4", {"learning_rate": 3e-4}),
        ("reinforce_03_lr_3e-3", {"learning_rate": 3e-3}),
        ("reinforce_04_gamma_0.95", {"gamma": 0.95}),
        ("reinforce_05_gamma_0.999", {"gamma": 0.999}),
        ("reinforce_06_entropy_0", {"entropy_coef": 0.0}),
        ("reinforce_07_entropy_0.05", {"entropy_coef": 0.05}),
        ("reinforce_08_net_64", {"hidden": [64, 64]}),
        ("reinforce_09_net_256", {"hidden": [256, 256]}),
        ("reinforce_10_no_return_norm", {"normalize_returns": False}),
        ("reinforce_11_value_baseline", {"value_baseline": True}),
        ("reinforce_12_batch_4_episodes", {"episodes_per_update": 4}),
    ],
    "ppo": [
        ("ppo_01_baseline", {}),
        ("ppo_02_lr_1e-4", {"learning_rate": 1e-4}),
        ("ppo_03_lr_1e-3", {"learning_rate": 1e-3}),
        ("ppo_04_clip_0.1", {"clip_epsilon": 0.1}),
        ("ppo_05_clip_0.3", {"clip_epsilon": 0.3}),
        ("ppo_06_entropy_0", {"entropy_coef": 0.0}),
        ("ppo_07_entropy_0.05", {"entropy_coef": 0.05}),
        ("ppo_08_rollout_512", {"rollout_steps": 512}),
        ("ppo_09_rollout_2048", {"rollout_steps": 2048}),
        ("ppo_10_epochs_10", {"epochs": 10}),
        ("ppo_11_minibatch_64", {"minibatch_size": 64}),
        ("ppo_12_minibatch_256", {"minibatch_size": 256}),
    ],
    "a2c": [
        ("a2c_01_baseline", {}),
        ("a2c_02_lr_3e-4", {"learning_rate": 3e-4}),
        ("a2c_03_lr_2e-3", {"learning_rate": 2e-3}),
        ("a2c_04_gamma_0.95", {"gamma": 0.95}),
        ("a2c_05_gamma_0.999", {"gamma": 0.999}),
        ("a2c_06_entropy_0", {"entropy_coef": 0.0}),
        ("a2c_07_entropy_0.05", {"entropy_coef": 0.05}),
        ("a2c_08_value_0.25", {"value_coef": 0.25}),
        ("a2c_09_value_1.0", {"value_coef": 1.0}),
        ("a2c_10_steps_5", {"rollout_steps": 5}),
        ("a2c_11_steps_64", {"rollout_steps": 64}),
        ("a2c_12_grad_clip_5", {"grad_clip": 5.0}),
        ("a2c_13_no_adv_norm", {"normalize_advantage": False}),
    ],
}
SUMMARY_COLUMNS = {
    "reinforce": ["learning_rate", "gamma", "entropy_coef", "hidden", "normalize_returns", "value_baseline", "episodes_per_update"],
    "ppo": ["learning_rate", "clip_epsilon", "entropy_coef", "rollout_steps", "epochs", "minibatch_size"],
    "a2c": ["learning_rate", "gamma", "entropy_coef", "value_coef", "rollout_steps", "grad_clip", "normalize_advantage"],
}
ALGORITHM_LABELS = {"reinforce": "REINFORCE", "ppo": "PPO", "a2c": "A2C"}


def build_actor(hidden):
    logits = TensorDictModule(
        MLP(in_features=OBS_SIZE, out_features=N_ACTIONS, num_cells=hidden, activation_class=nn.Tanh),
        in_keys=["observation"],
        out_keys=["logits"],
    )
    return ProbabilisticActor(
        logits,
        in_keys={"logits": "logits", "mask": "action_mask"},
        out_keys=["action"],
        distribution_class=MaskedCategorical,
        return_log_prob=True,
    )


def build_critic(hidden):
    return ValueOperator(
        MLP(in_features=OBS_SIZE, out_features=1, num_cells=hidden, activation_class=nn.Tanh),
        in_keys=["observation"],
    )


def greedy_action(actor):
    def select(td):
        return actor.get_dist(td.clone()).probs.argmax(-1).item()

    return select


def load_policy(path):
    checkpoint = torch.load(path, map_location="cpu")
    actor = build_actor(checkpoint["config"]["hidden"])
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    return greedy_action(actor)


class Collector:
    def __init__(self, env, actor, run_seed, episode_budget):
        self.env = env
        self.actor = actor
        self.run_seed = run_seed
        self.episode_budget = episode_budget
        self.episodes = 0
        self.steps = 0
        self.td = None
        self.ep_return = 0.0
        self.ep_length = 0

    @property
    def finished(self):
        return self.episodes >= self.episode_budget

    def collect(self, max_steps=None, max_episodes=None):
        transitions, completed = [], []
        while not self.finished:
            if self.td is None:
                self.td = prepare(self.env.reset(seed=train_seed(self.run_seed, self.episodes)))
            with torch.no_grad():
                self.actor(self.td)
            td = prepare(self.env.step(self.td))
            reward = td["next", "reward"].item()
            self.ep_return += reward
            self.ep_length += 1
            self.steps += 1
            transition = td.select(*TRANSITION_KEYS, "sample_log_prob").clone()
            transition["next", "reward"] = transition["next", "reward"] * REWARD_SCALE
            transitions.append(transition)
            if td["next", "done"].item():
                completed.append((self.ep_return, self.ep_length))
                self.episodes += 1
                self.td, self.ep_return, self.ep_length = None, 0.0, 0
                if max_episodes is not None and len(completed) >= max_episodes:
                    break
            else:
                self.td = step_mdp(td)
            if max_steps is not None and len(transitions) >= max_steps:
                break
        return torch.stack(transitions), completed


def discounted_returns(batch, gamma):
    rewards = batch["next", "reward"].squeeze(-1)
    dones = batch["next", "done"].squeeze(-1)
    returns = torch.zeros_like(rewards)
    running = 0.0
    for t in reversed(range(rewards.shape[0])):
        running = rewards[t] + gamma * running * (~dones[t]).float()
        returns[t] = running
    return returns


class Trainer:
    def __init__(self, algorithm, name, config, episodes, verbose):
        self.algorithm = algorithm
        self.label = ALGORITHM_LABELS[algorithm]
        self.name = name
        self.config = config
        self.episodes = episodes
        self.verbose = verbose
        self.actor = build_actor(config["hidden"])
        self.critic = build_critic(config["hidden"])
        self.collector = Collector(make_env(), self.actor, config["seed"], episodes)
        self.history = {
            "episode_return": [],
            "episode_length": [],
            "update_episode": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "validation": [],
        }
        self.best_return = -np.inf
        self.best_state = None
        self.next_validation = VALIDATE_EVERY
        self.start = time.time()

    def record_episodes(self, completed):
        for ep_return, ep_length in completed:
            self.history["episode_return"].append(ep_return)
            self.history["episode_length"].append(ep_length)

    def record_update(self, policy_loss, value_loss, entropy):
        self.history["update_episode"].append(self.collector.episodes)
        self.history["policy_loss"].append(float(policy_loss))
        self.history["value_loss"].append(None if value_loss is None else float(value_loss))
        self.history["entropy"].append(float(entropy))

    def maybe_validate(self):
        done = self.collector.episodes
        if done < self.next_validation and not self.collector.finished:
            return
        while self.next_validation <= done:
            self.next_validation += VALIDATE_EVERY
        score = validate(greedy_action(self.actor))
        self.history["validation"].append({"episode": done, "return": score, "steps": self.collector.steps})
        if score > self.best_return:
            self.best_return = score
            self.best_state = {"actor": copy.deepcopy(self.actor.state_dict()), "critic": copy.deepcopy(self.critic.state_dict())}
        if self.verbose:
            last_return = self.history["episode_return"][-1] if self.history["episode_return"] else 0.0
            entropy = self.history["entropy"][-1] if self.history["entropy"] else float("nan")
            log_progress(self.label, self.name, done - 1, self.episodes, last_return, f"entropy {entropy:.3f} val {score:8.1f}", self.start)

    def finish(self):
        self.actor.load_state_dict(self.best_state["actor"])
        self.critic.load_state_dict(self.best_state["critic"])
        train_seconds = time.time() - self.start
        MODELS_DIR.joinpath("pg").mkdir(parents=True, exist_ok=True)
        torch.save(
            {"algorithm": self.algorithm, "config": self.config, **self.best_state},
            MODELS_DIR / "pg" / f"{self.name}.pt",
        )
        record = {
            "algorithm": self.label,
            "name": self.name,
            "config": self.config,
            "episodes": self.episodes,
            "total_steps": self.collector.steps,
            "train_seconds": train_seconds,
            "best_validation_return": self.best_return,
            "history": self.history,
            "evaluation": final_evaluation(greedy_action(self.actor)),
        }
        save_run(self.algorithm, self.name, record)
        if self.verbose:
            std = record["evaluation"]["standard"]
            print(
                f"[{self.label}:{self.name}] done in {train_seconds:.0f}s | eval return {std['return_mean']:.1f} "
                f"on-time {std['on_time_rate_mean']:.1%}",
                flush=True,
            )
        return record


def train_reinforce(trainer):
    config = trainer.config
    params = list(trainer.actor.parameters()) + (list(trainer.critic.parameters()) if config["value_baseline"] else [])
    optimizer = torch.optim.Adam(params, lr=config["learning_rate"])
    while not trainer.collector.finished:
        batch, completed = trainer.collector.collect(max_episodes=config["episodes_per_update"])
        trainer.record_episodes(completed)
        returns = discounted_returns(batch, config["gamma"])
        advantage = returns
        value_loss = None
        if config["value_baseline"]:
            values = trainer.critic(batch.select("observation").clone())["state_value"].squeeze(-1)
            value_loss = nn.functional.smooth_l1_loss(values, returns)
            advantage = returns - values.detach()
        if config["normalize_returns"] and advantage.numel() > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        dist = trainer.actor.get_dist(batch.select("observation", "action_mask").clone())
        log_prob = dist.log_prob(batch["action"])
        entropy = dist.entropy().mean()
        policy_loss = -(log_prob * advantage).mean()
        loss = policy_loss - config["entropy_coef"] * entropy + (value_loss if value_loss is not None else 0.0)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, config["grad_clip"])
        optimizer.step()
        trainer.record_update(policy_loss.item(), None if value_loss is None else value_loss.item(), entropy.item())
        trainer.maybe_validate()


def train_actor_critic(trainer, loss_module, epochs, minibatch_size):
    config = trainer.config
    advantage_module = GAE(gamma=config["gamma"], lmbda=config["gae_lambda"], value_network=trainer.critic, average_gae=False)
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=config["learning_rate"])
    while not trainer.collector.finished:
        batch, completed = trainer.collector.collect(max_steps=config["rollout_steps"])
        trainer.record_episodes(completed)
        with torch.no_grad():
            advantage_module(batch)
            if config.get("normalize_advantage") and batch.shape[0] > 1:
                advantage = batch["advantage"]
                batch["advantage"] = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        size = batch.shape[0]
        stats = []
        for _ in range(epochs):
            for idx in torch.randperm(size).split(min(minibatch_size, size)):
                minibatch = batch[idx].clone()
                out = loss_module(minibatch)
                loss = out["loss_objective"] + out["loss_critic"] + out.get("loss_entropy", torch.zeros(()))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), config["grad_clip"])
                optimizer.step()
                if "entropy" in out.keys():
                    entropy = out["entropy"].item()
                else:
                    with torch.no_grad():
                        entropy = trainer.actor.get_dist(minibatch.select("observation", "action_mask")).entropy().mean().item()
                stats.append((out["loss_objective"].item(), out["loss_critic"].item(), entropy))
        policy_loss, value_loss, entropy = np.mean(stats, axis=0)
        trainer.record_update(policy_loss, value_loss, entropy)
        trainer.maybe_validate()


def train(algorithm, experiment, episodes=TRAIN_EPISODES, verbose=True):
    name, overrides = experiment
    config = {**BASE_CONFIGS[algorithm], **overrides}
    torch.set_num_threads(1)
    set_seeds(config["seed"])
    trainer = Trainer(algorithm, name, config, episodes, verbose)
    if algorithm == "reinforce":
        train_reinforce(trainer)
    elif algorithm == "ppo":
        loss_module = ClipPPOLoss(
            trainer.actor,
            trainer.critic,
            clip_epsilon=config["clip_epsilon"],
            entropy_bonus=True,
            entropy_coef=config["entropy_coef"],
            critic_coef=config["value_coef"],
            loss_critic_type="smooth_l1",
            normalize_advantage=True,
        )
        train_actor_critic(trainer, loss_module, config["epochs"], config["minibatch_size"])
    else:
        loss_module = A2CLoss(
            trainer.actor,
            trainer.critic,
            entropy_bonus=True,
            entropy_coef=config["entropy_coef"],
            critic_coef=config["value_coef"],
            loss_critic_type="smooth_l1",
        )
        train_actor_critic(trainer, loss_module, 1, config["rollout_steps"])
    return trainer.finish()


def run_experiment(job, episodes):
    algorithm, experiment = job
    return train(algorithm, experiment, episodes)


def main():
    parser = argparse.ArgumentParser(description="Train REINFORCE, PPO and A2C agents (TorchRL) on the maternal referral environment.")
    parser.add_argument("--algorithms", nargs="*", default=list(EXPERIMENTS), choices=list(EXPERIMENTS))
    parser.add_argument("--experiments", nargs="*", default=None, help="experiment names to run (default: all)")
    parser.add_argument("--episodes", type=int, default=TRAIN_EPISODES)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--summarize", action="store_true", help="rebuild summary tables from saved run logs")
    args = parser.parse_args()
    if args.summarize:
        for algorithm in args.algorithms:
            records = [json.loads((LOGS_DIR / algorithm / f"{name}.json").read_text()) for name, _ in EXPERIMENTS[algorithm]]
            write_summary(algorithm, records, SUMMARY_COLUMNS[algorithm])
        return
    jobs = [
        (algorithm, experiment)
        for algorithm in args.algorithms
        for experiment in EXPERIMENTS[algorithm]
        if args.experiments is None or experiment[0] in args.experiments
    ]
    records = run_many(functools.partial(run_experiment, episodes=args.episodes), jobs, args.workers)
    if args.experiments is None:
        for algorithm in args.algorithms:
            write_summary(algorithm, [r for r in records if r["name"].startswith(algorithm)], SUMMARY_COLUMNS[algorithm])


if __name__ == "__main__":
    main()
