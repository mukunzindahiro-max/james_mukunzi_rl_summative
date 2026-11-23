import argparse
import json
from pathlib import Path

import numpy as np

from environment.custom_env import (
    HEALTH_CENTRE_NODES,
    HOSPITALS,
    WAIT_ACTION,
    MaternalReferralEnv,
    decode_action,
)

EVAL_SEEDS = list(range(10_000, 10_020))
SCENARIOS = {
    "standard": {},
    "rainy_season": {"rain_prob": 0.8, "general_duty_per_day": 3.0},
}
EPISODE_METRICS = [
    "return",
    "length",
    "completed",
    "completed_on_time",
    "on_time_rate",
    "unresolved",
    "mean_wait_minutes",
    "mean_referral_minutes",
    "ambulance_utilization",
    "total_km",
    "invalid_actions",
]
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"


def random_policy(seed=0):
    rng = np.random.default_rng(seed)

    def act(env, obs):
        return int(rng.choice(np.flatnonzero(obs["action_mask"])))

    return act


def nearest_policy():
    def act(env, obs):
        referral = env._current_referral()
        if referral is None:
            return WAIT_ACTION
        minutes, _, _ = env._table()
        origin = HEALTH_CENTRE_NODES[referral["origin"]]
        best, best_time = WAIT_ACTION, np.inf
        for action in np.flatnonzero(obs["action_mask"]):
            if action == WAIT_ACTION:
                continue
            ambulance, hospital = decode_action(int(action))
            total = minutes[env.ambulances[ambulance]["base"], origin] + minutes[origin, HOSPITALS[hospital]["node"]]
            if total < best_time:
                best, best_time = int(action), total
        return best

    return act


def run_episode(policy, seed, scenario="standard", env=None):
    env = env or MaternalReferralEnv(**SCENARIOS[scenario])
    obs, _ = env.reset(seed=seed)
    total, length, invalid, done = 0.0, 0, 0, False
    while not done:
        obs, reward, terminated, truncated, info = env.step(policy(env, obs))
        total += reward
        length += 1
        invalid += info["outcome"] not in ("dispatch", "wait")
        done = terminated or truncated
    metrics = env.metrics()
    return {
        "seed": seed,
        "return": total,
        "length": length,
        "invalid_actions": invalid,
        **{k: v for k, v in metrics.items() if k in EPISODE_METRICS},
        "hospital_admissions": metrics["hospital_admissions"],
    }


def summarise(episodes):
    summary = {}
    for key in EPISODE_METRICS:
        values = np.array([e[key] for e in episodes], dtype=np.float64)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std())
    summary["hospital_admissions"] = np.sum([e["hospital_admissions"] for e in episodes], axis=0).tolist()
    return summary


def evaluate(policy, scenario="standard", seeds=None):
    seeds = EVAL_SEEDS if seeds is None else seeds
    env = MaternalReferralEnv(**SCENARIOS[scenario])
    episodes = [run_episode(policy, seed, scenario, env) for seed in seeds]
    return {"scenario": scenario, "seeds": list(seeds), "summary": summarise(episodes), "episodes": episodes}


def format_row(name, scenario, summary):
    return (
        f"{name:<10} {scenario:<13} "
        f"{summary['return_mean']:>8.1f} ± {summary['return_std']:<6.1f} "
        f"{summary['on_time_rate_mean']:>7.1%} {summary['unresolved_mean']:>10.2f} "
        f"{summary['mean_wait_minutes_mean']:>9.1f} {summary['mean_referral_minutes_mean']:>9.1f} "
        f"{summary['ambulance_utilization_mean']:>6.1%} {summary['total_km_mean']:>7.0f}"
    )


def table_header():
    return (
        f"{'policy':<10} {'scenario':<13} {'return':>17} {'on-time':>7} {'unresolved':>10} "
        f"{'wait(min)':>9} {'trip(min)':>9} {'util':>6} {'km':>7}"
    )


def run_baselines(output=LOGS_DIR / "baselines.json"):
    results = {}
    print(table_header())
    for name, factory in (("random", lambda: random_policy(0)), ("nearest", nearest_policy)):
        results[name] = {}
        for scenario in SCENARIOS:
            result = evaluate(factory(), scenario)
            results[name][scenario] = result
            print(format_row(name, scenario, result["summary"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(f"Saved baseline evaluation to {output}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate baseline dispatch policies on held-out seeds.")
    parser.add_argument("--output", type=Path, default=LOGS_DIR / "baselines.json")
    run_baselines(parser.parse_args().output)
