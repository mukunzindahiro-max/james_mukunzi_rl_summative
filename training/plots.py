import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba

from training.common import LOGS_DIR, TRAIN_EPISODES

ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT / "assets"
ALGORITHMS = {"dqn": "DQN", "reinforce": "REINFORCE", "ppo": "PPO", "a2c": "A2C"}
COLOURS = {"dqn": "#2563eb", "reinforce": "#e11d48", "ppo": "#059669", "a2c": "#f59e0b"}
BASELINE_COLOURS = {"random": "#a855f7", "nearest": "#0891b2"}
ENTROPY_COLOURS = {0.0: "#e11d48", 0.01: "#2563eb", 0.05: "#059669"}
PG = ["reinforce", "ppo", "a2c"]
INK = "#1e293b"
STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "#f8fafc",
    "axes.edgecolor": "#cbd5e1",
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e2e8f0",
    "grid.linewidth": 0.9,
    "xtick.color": "#475569",
    "ytick.color": "#475569",
    "font.size": 10.5,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#e2e8f0",
    "lines.linewidth": 2.2,
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
}


def load(algorithm, name):
    return json.loads((LOGS_DIR / algorithm / f"{name}.json").read_text())


def smooth(values, window=50):
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.concatenate([np.full(window - 1, values[0]), values])
    return np.convolve(padded, kernel, mode="valid")


def per_episode(update_episode, values):
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


def baseline_lines(ax, comparison, horizontal=True):
    for name, style in (("nearest", "--"), ("random", ":")):
        value = comparison["baselines"][name]["standard"]["return_mean"]
        line = ax.axhline if horizontal else ax.axvline
        line(value, color=BASELINE_COLOURS[name], linestyle=style, linewidth=1.8, label=f"{name.capitalize()} baseline ({value:.0f})")


def save(fig, name, rect=(0, 0, 1, 1)):
    fig.tight_layout(rect=rect)
    fig.savefig(ASSETS_DIR / name, dpi=170)
    plt.close(fig)
    print(f"Saved assets/{name}")


def plot_training_rewards(comparison):
    fig, ax = plt.subplots(figsize=(11, 5.8))
    for algorithm, label in ALGORITHMS.items():
        curves = np.array([smooth(r["history"]["episode_return"]) for r in seed_records(comparison, algorithm)])
        x = np.arange(1, curves.shape[1] + 1)
        mean, std = curves.mean(0), curves.std(0)
        ax.plot(x, mean, color=COLOURS[algorithm], label=f"{label} ({comparison['algorithms'][algorithm]['best_experiment']})")
        ax.fill_between(x, mean - std, mean + std, color=COLOURS[algorithm], alpha=0.14, linewidth=0)
    baseline_lines(ax, comparison)
    ax.set_title("Training episode return (best configuration, mean ± std over 3 seeds, 50-episode moving average)")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Episode return")
    ax.legend(loc="lower right", fontsize=9)
    save(fig, "cumulative_rewards.png")


def plot_dqn_objective(comparison):
    record = load("dqn", comparison["algorithms"]["dqn"]["best_experiment"])
    history = record["history"]
    episodes = np.arange(1, len(history["loss"]) + 1)
    mask = np.array([v is not None for v in history["loss"]])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    loss = np.array([v for v in history["loss"] if v is not None])
    axes[0].plot(episodes[mask], loss, color=COLOURS["dqn"], alpha=0.25, linewidth=1, label="per episode")
    axes[0].plot(episodes[mask], smooth(loss), color=COLOURS["dqn"], label="50-episode average")
    axes[0].set_title("TD loss (smooth L1)")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=9)
    q = np.array([v for v in history["q_value"] if v is not None])
    axes[1].plot(episodes[mask], q, color="#7c3aed", alpha=0.25, linewidth=1)
    axes[1].plot(episodes[mask], smooth(q), color="#7c3aed")
    axes[1].set_title("Mean predicted Q-value")
    axes[2].plot(episodes, history["epsilon"], color="#ea580c")
    axes[2].fill_between(episodes, history["epsilon"], color="#ea580c", alpha=0.12, linewidth=0)
    axes[2].set_title("Exploration rate ε")
    for ax in axes:
        ax.set_xlabel("Training episode")
    fig.suptitle(f"DQN objective curves ({record['name']})")
    save(fig, "dqn_objective.png")


def plot_pg_entropy(comparison):
    fig, ax = plt.subplots(figsize=(11, 5.4))
    for algorithm in PG:
        record = load(algorithm, comparison["algorithms"][algorithm]["best_experiment"])
        x, y = per_episode(record["history"]["update_episode"], record["history"]["entropy"])
        ax.plot(x, smooth(y, 10 if algorithm == "ppo" else 50), color=COLOURS[algorithm], label=f"{ALGORITHMS[algorithm]} ({record['name']})")
    ax.axhline(np.log(16), color="#7c3aed", linestyle=":", linewidth=1.8, label="Uniform over 16 actions (ln 16)")
    ax.set_title("Policy entropy during training (best configuration per policy-gradient method)")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Mean policy entropy (nats, masked actions)")
    ax.legend(fontsize=9)
    save(fig, "pg_entropy.png")


def plot_entropy_sweep():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, algorithm in zip(axes, PG):
        frame = pd.read_csv(LOGS_DIR / algorithm / "summary.csv")
        for name in frame["name"]:
            if not (name.endswith("_01_baseline") or "entropy" in name):
                continue
            record = load(algorithm, name)
            coef = record["config"]["entropy_coef"]
            x, y = per_episode(record["history"]["update_episode"], record["history"]["entropy"])
            ax.plot(x, smooth(y, 10 if algorithm == "ppo" else 50), color=ENTROPY_COLOURS[coef], label=f"entropy coef {coef}")
        ax.set_title(ALGORITHMS[algorithm], color=COLOURS[algorithm])
        ax.set_xlabel("Training episode")
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Policy entropy (nats)")
    fig.suptitle("Effect of the entropy coefficient on policy entropy")
    save(fig, "pg_entropy_sweep.png")


def plot_convergence(comparison):
    fig, ax = plt.subplots(figsize=(11, 5.6))
    for algorithm, label in ALGORITHMS.items():
        records = seed_records(comparison, algorithm)
        steps = np.array([[v["steps"] for v in r["history"]["validation"]] for r in records]).mean(0)
        returns = np.array([[v["return"] for v in r["history"]["validation"]] for r in records])
        mean, std = returns.mean(0), returns.std(0)
        ax.plot(steps, mean, marker="o", markersize=3.5, color=COLOURS[algorithm], label=label)
        ax.fill_between(steps, mean - std, mean + std, color=COLOURS[algorithm], alpha=0.14, linewidth=0)
    baseline_lines(ax, comparison)
    ax.set_title("Convergence: greedy validation return vs environment steps (mean ± std over 3 seeds)")
    ax.set_xlabel("Environment steps (dispatch decisions)")
    ax.set_ylabel("Validation return (seeds 5000-5004)")
    ax.legend(fontsize=9, loc="lower right")
    save(fig, "convergence.png")


def plot_generalization(comparison):
    names = ["random", "nearest", *ALGORITHMS]
    labels = ["Random", "Nearest", *ALGORITHMS.values()]
    entries = [comparison["baselines"][n] if n in BASELINE_COLOURS else comparison["algorithms"][n]["evaluation"] for n in names]
    colours = [BASELINE_COLOURS.get(n) or COLOURS[n] for n in names]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    for offset, scenario, alpha in ((-0.2, "standard", 1.0), (0.2, "rainy_season", 0.45)):
        faces = [to_rgba(c, alpha) for c in colours]
        returns = [e[scenario]["return_mean"] for e in entries]
        on_time = [100 * e[scenario]["on_time_rate"] for e in entries]
        bars = axes[0].bar(x + offset, returns, 0.38, yerr=[e[scenario]["return_std"] for e in entries], color=faces,
                           edgecolor=colours, linewidth=1.4, capsize=3, error_kw={"ecolor": "#475569", "elinewidth": 1})
        axes[0].bar_label(bars, labels=[f"{v:.0f}" for v in returns], padding=14, fontsize=8, color=INK)
        bars = axes[1].bar(x + offset, on_time, 0.38, color=faces, edgecolor=colours, linewidth=1.4)
        axes[1].bar_label(bars, labels=[f"{v:.0f}" for v in on_time], padding=2, fontsize=8, color=INK)
    axes[0].set_ylabel("Return (mean ± std, 20 held-out seeds)")
    axes[0].set_title("Held-out evaluation return")
    axes[0].set_ylim(0, 2050)
    axes[1].set_ylabel("Referrals reaching hospital on time (%)")
    axes[1].set_title("Held-out on-time rate")
    axes[1].set_ylim(60, 101)
    legend = [plt.Rectangle((0, 0), 1, 1, facecolor=to_rgba(INK, a), edgecolor=INK) for a in (1.0, 0.45)]
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.grid(axis="x", visible=False)
    fig.legend(legend, ["Standard scenario (left bar)", "Rainy season (right bar)"], fontsize=10, loc="lower center", ncol=2)
    fig.suptitle("Generalization to unseen seeds (10000-10019) and to the rainy-season scenario")
    save(fig, "generalization.png", rect=(0, 0.07, 1, 1))


def plot_hyperparameter_sweeps(comparison):
    nearest = comparison["baselines"]["nearest"]["standard"]["return_mean"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, (algorithm, label) in zip(axes.flat, ALGORITHMS.items()):
        frame = pd.read_csv(LOGS_DIR / algorithm / "summary.csv")
        values = frame["eval_return"].to_numpy()
        best = frame["val_best"].idxmax()
        typical = np.median(values)
        low = max(values[values > typical - 500].min() - 80, 0)
        high = max(values.max(), nearest if nearest - values.max() < 200 else values.max()) + 70
        colour = COLOURS[algorithm]
        faces = [to_rgba(colour, 1.0 if i == best else 0.35) for i in range(len(frame))]
        y = np.arange(len(frame))
        ax.barh(y, np.maximum(values, low) - low, left=low, color=faces, edgecolor=colour, linewidth=1.2, height=0.72)
        for yi, v in zip(y, values):
            if v < low:
                ax.text(low + 8, yi, f"{v:.0f}  (collapsed)", va="center", fontsize=8.5, color=COLOURS["reinforce"], fontweight="bold")
            else:
                ax.text(v + 6, yi, f"{v:.0f}" + ("  ★ selected" if yi == best else ""), va="center", fontsize=8.5,
                        color=INK, fontweight="bold" if yi == best else "normal")
        if low < nearest < high:
            ax.axvline(nearest, color=BASELINE_COLOURS["nearest"], linestyle="--", linewidth=1.6, label=f"Nearest baseline ({nearest:.0f})")
            ax.legend(fontsize=8.5, loc="lower right")
        ax.set_yticks(y, [n.split("_", 2)[-1].replace("_", " ") for n in frame["name"]], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(low, high + 90)
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{label}: {len(frame)} configurations", color=colour)
        ax.set_xlabel("Held-out evaluation return")
    fig.suptitle(f"Hyperparameter sweeps (each run trained for {TRAIN_EPISODES} episodes on the same environment)")
    save(fig, "hyperparameter_sweeps.png")


def plot_seed_robustness(comparison):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, (algorithm, label) in enumerate(ALGORITHMS.items()):
        returns = comparison["algorithms"][algorithm]["seed_robustness"]["eval_returns"]
        ax.scatter([i] * len(returns), returns, s=70, color=COLOURS[algorithm], edgecolor="white", linewidth=1.2, zorder=3)
        ax.errorbar(i + 0.18, np.mean(returns), yerr=np.std(returns), fmt="D", markersize=7, color=COLOURS[algorithm],
                    markeredgecolor=INK, ecolor=INK, capsize=5, zorder=3)
        ax.text(i + 0.3, np.mean(returns), f"{np.mean(returns):.0f} ± {np.std(returns):.0f}", va="center", fontsize=9, color=INK)
    baseline_lines(ax, comparison)
    ax.set_xticks(range(len(ALGORITHMS)), ALGORITHMS.values())
    ax.set_xlim(-0.4, len(ALGORITHMS) - 0.1)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Held-out return")
    ax.set_title("Seed robustness of the best configuration (training seeds 1, 2, 3)")
    ax.legend(fontsize=8.5, loc="lower left")
    save(fig, "seed_robustness.png")


def main():
    ASSETS_DIR.mkdir(exist_ok=True)
    comparison = json.loads((LOGS_DIR / "comparison.json").read_text())
    plt.rcParams.update(STYLE)
    plot_training_rewards(comparison)
    plot_dqn_objective(comparison)
    plot_pg_entropy(comparison)
    plot_entropy_sweep()
    plot_convergence(comparison)
    plot_generalization(comparison)
    plot_hyperparameter_sweeps(comparison)
    plot_seed_robustness(comparison)


if __name__ == "__main__":
    main()
