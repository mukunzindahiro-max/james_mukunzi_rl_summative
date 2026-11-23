import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from training.common import LOGS_DIR, TRAIN_EPISODES

ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT / "assets"
ALGORITHMS = {"dqn": "DQN", "reinforce": "REINFORCE", "ppo": "PPO", "a2c": "A2C"}
COLOURS = {"dqn": "#1f77b4", "reinforce": "#d62728", "ppo": "#2ca02c", "a2c": "#ff7f0e"}
PG = ["reinforce", "ppo", "a2c"]


def load(algorithm, name):
    return json.loads((LOGS_DIR / algorithm / f"{name}.json").read_text())


def smooth(values, window=50):
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.concatenate([np.full(window - 1, values[0]), values])
    return np.convolve(padded, kernel, mode="valid")


def per_episode(update_episode, values, episodes):
    groups = defaultdict(list)
    for e, v in zip(update_episode, values):
        if v is not None:
            groups[e].append(v)
    xs = sorted(groups)
    return np.array(xs) + 1, np.array([np.mean(groups[x]) for x in xs])


def seed_records(comparison, algorithm):
    best = comparison["algorithms"][algorithm]["best_experiment"]
    names = [best] + [f"{best}_seed{s}" for s in comparison["algorithms"][algorithm]["seed_robustness"]["seeds"][1:]]
    return [load(algorithm, n) for n in names]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / name, dpi=160)
    plt.close(fig)
    print(f"Saved assets/{name}")


def plot_training_rewards(comparison):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for algorithm, label in ALGORITHMS.items():
        curves = np.array([smooth(r["history"]["episode_return"]) for r in seed_records(comparison, algorithm)])
        x = np.arange(1, curves.shape[1] + 1)
        mean, std = curves.mean(0), curves.std(0)
        ax.plot(x, mean, color=COLOURS[algorithm], label=f"{label} ({comparison['algorithms'][algorithm]['best_experiment']})")
        ax.fill_between(x, mean - std, mean + std, color=COLOURS[algorithm], alpha=0.15)
    for name, style in (("nearest", "--"), ("random", ":")):
        ax.axhline(comparison["baselines"][name]["standard"]["return_mean"], color="black", linestyle=style, linewidth=1.2, label=f"{name.capitalize()} baseline (held-out)")
    ax.set_title("Training episode return of the best configuration per algorithm (mean ± std over 3 seeds, 50-episode moving average)")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Episode return")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.title.set_fontsize(10)
    save(fig, "cumulative_rewards.png")


def plot_dqn_objective(comparison):
    record = load("dqn", comparison["algorithms"]["dqn"]["best_experiment"])
    history = record["history"]
    episodes = np.arange(1, len(history["loss"]) + 1)
    mask = np.array([v is not None for v in history["loss"]])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    loss = np.array([v for v in history["loss"] if v is not None])
    axes[0].plot(episodes[mask], loss, color=COLOURS["dqn"], alpha=0.3, label="per episode")
    axes[0].plot(episodes[mask], smooth(loss), color=COLOURS["dqn"], label="50-episode average")
    axes[0].set_title("TD loss (smooth L1)")
    axes[0].set_yscale("log")
    q = np.array([v for v in history["q_value"] if v is not None])
    axes[1].plot(episodes[mask], q, color="#9467bd", alpha=0.3)
    axes[1].plot(episodes[mask], smooth(q), color="#9467bd")
    axes[1].set_title("Mean predicted Q-value")
    axes[2].plot(episodes, history["epsilon"], color="#8c564b")
    axes[2].set_title("Exploration rate ε")
    for ax in axes:
        ax.set_xlabel("Training episode")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"DQN objective curves ({record['name']})", fontsize=11)
    save(fig, "dqn_objective.png")


def plot_pg_entropy(comparison):
    fig, ax = plt.subplots(figsize=(10, 5))
    for algorithm in PG:
        record = load(algorithm, comparison["algorithms"][algorithm]["best_experiment"])
        h = record["history"]
        x, y = per_episode(h["update_episode"], h["entropy"], record["episodes"])
        ax.plot(x, smooth(y, 10 if algorithm == "ppo" else 50), color=COLOURS[algorithm], label=f"{ALGORITHMS[algorithm]} ({record['name']})")
    ax.axhline(np.log(16), color="grey", linestyle=":", label="Uniform over 16 actions (ln 16)")
    ax.set_title("Policy entropy during training (best configuration per policy-gradient method)")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Mean policy entropy (nats, masked actions)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    save(fig, "pg_entropy.png")


def plot_entropy_sweep():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, algorithm in zip(axes, PG):
        frame = pd.read_csv(LOGS_DIR / algorithm / "summary.csv")
        for name in frame["name"]:
            record = load(algorithm, name)
            coef = record["config"]["entropy_coef"]
            if name.endswith("_01_baseline") or "entropy" in name:
                h = record["history"]
                x, y = per_episode(h["update_episode"], h["entropy"], record["episodes"])
                ax.plot(x, smooth(y, 10 if algorithm == "ppo" else 50), label=f"entropy coef {coef}")
        ax.set_title(ALGORITHMS[algorithm])
        ax.set_xlabel("Training episode")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Policy entropy (nats)")
    fig.suptitle("Effect of the entropy coefficient on policy entropy", fontsize=11)
    save(fig, "pg_entropy_sweep.png")


def plot_convergence(comparison):
    fig, ax = plt.subplots(figsize=(10, 5))
    for algorithm, label in ALGORITHMS.items():
        records = seed_records(comparison, algorithm)
        steps = np.array([[v["steps"] for v in r["history"]["validation"]] for r in records]).mean(0)
        returns = np.array([[v["return"] for v in r["history"]["validation"]] for r in records])
        mean, std = returns.mean(0), returns.std(0)
        ax.plot(steps, mean, marker="o", markersize=3, color=COLOURS[algorithm], label=label)
        ax.fill_between(steps, mean - std, mean + std, color=COLOURS[algorithm], alpha=0.15)
    ax.axhline(comparison["baselines"]["nearest"]["standard"]["return_mean"], color="black", linestyle="--", linewidth=1.2, label="Nearest baseline")
    ax.axhline(comparison["baselines"]["random"]["standard"]["return_mean"], color="black", linestyle=":", linewidth=1.2, label="Random baseline")
    ax.set_title("Convergence: greedy validation return (seeds 5000-5004) vs environment steps, mean ± std over 3 seeds")
    ax.title.set_fontsize(10)
    ax.set_xlabel("Environment steps (dispatch decisions)")
    ax.set_ylabel("Validation return")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    save(fig, "convergence.png")


def plot_generalization(comparison):
    names = ["random", "nearest", *ALGORITHMS]
    labels = ["Random", "Nearest", *ALGORITHMS.values()]
    entries = [comparison["baselines"][n] if n in ("random", "nearest") else comparison["algorithms"][n]["evaluation"] for n in names]
    colours = ["#7f7f7f", "#404040", *COLOURS.values()]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for offset, scenario, hatch in ((-0.2, "standard", None), (0.2, "rainy_season", "//")):
        axes[0].bar(x + offset, [e[scenario]["return_mean"] for e in entries], 0.4, yerr=[e[scenario]["return_std"] for e in entries],
                    color=colours, hatch=hatch, edgecolor="white", capsize=3, label=scenario.replace("_", " "))
        axes[1].bar(x + offset, [100 * e[scenario]["on_time_rate"] for e in entries], 0.4, color=colours, hatch=hatch, edgecolor="white", label=scenario.replace("_", " "))
    axes[0].set_ylabel("Return (mean ± std, 20 held-out seeds)")
    axes[0].set_title("Held-out evaluation return")
    axes[1].set_ylabel("Referrals reaching hospital on time (%)")
    axes[1].set_title("Held-out on-time rate")
    axes[1].set_ylim(60, 100)
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Generalization to unseen seeds 10000-10019 and to the rainy-season scenario (not used for training)", fontsize=11)
    save(fig, "generalization.png")


def plot_hyperparameter_sweeps():
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (algorithm, label) in zip(axes.flat, ALGORITHMS.items()):
        frame = pd.read_csv(LOGS_DIR / algorithm / "summary.csv")
        best = frame["val_best"].idxmax()
        colours = [COLOURS[algorithm] if i == best else "#b0b0b0" for i in range(len(frame))]
        y = np.arange(len(frame))
        ax.barh(y, frame["eval_return"], color=colours)
        ax.set_yticks(y, [n.split("_", 2)[-1] for n in frame["name"]], fontsize=8)
        ax.invert_yaxis()
        low = frame["eval_return"].min()
        ax.set_xlim(low - 100, frame["eval_return"].max() + 60)
        for yi, v in zip(y, frame["eval_return"]):
            ax.text(v + 5, yi, f"{v:.0f}", va="center", fontsize=7)
        ax.set_title(f"{label}: {len(frame)} configurations (coloured = selected by validation)", fontsize=10)
        ax.set_xlabel("Held-out evaluation return")
        ax.grid(alpha=0.3, axis="x")
    fig.suptitle(f"Hyperparameter sweeps (each run trained for {TRAIN_EPISODES} episodes on the same environment)", fontsize=12)
    save(fig, "hyperparameter_sweeps.png")


def plot_seed_robustness(comparison):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, (algorithm, label) in enumerate(ALGORITHMS.items()):
        returns = comparison["algorithms"][algorithm]["seed_robustness"]["eval_returns"]
        ax.scatter([i] * len(returns), returns, color=COLOURS[algorithm], zorder=3)
        ax.errorbar(i + 0.15, np.mean(returns), yerr=np.std(returns), fmt="D", color="black", capsize=4)
    ax.axhline(comparison["baselines"]["nearest"]["standard"]["return_mean"], color="black", linestyle="--", linewidth=1, label="Nearest baseline")
    ax.set_xticks(range(len(ALGORITHMS)), ALGORITHMS.values())
    ax.set_ylabel("Held-out return")
    ax.set_title("Seed robustness of the best configuration (training seeds 1, 2, 3)", fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    save(fig, "seed_robustness.png")


def main():
    ASSETS_DIR.mkdir(exist_ok=True)
    comparison = json.loads((LOGS_DIR / "comparison.json").read_text())
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    plot_training_rewards(comparison)
    plot_dqn_objective(comparison)
    plot_pg_entropy(comparison)
    plot_entropy_sweep()
    plot_convergence(comparison)
    plot_generalization(comparison)
    plot_hyperparameter_sweeps()
    plot_seed_robustness(comparison)


if __name__ == "__main__":
    main()
