import argparse
import functools
import json

import numpy as np

from training import dqn_training, pg_training
from training.common import LOGS_DIR, MODELS_DIR, TRAIN_EPISODES, run_many

ALGORITHMS = {
    "dqn": ("DQN", dqn_training.EXPERIMENTS, "dqn"),
    "reinforce": ("REINFORCE", pg_training.EXPERIMENTS["reinforce"], "pg"),
    "ppo": ("PPO", pg_training.EXPERIMENTS["ppo"], "pg"),
    "a2c": ("A2C", pg_training.EXPERIMENTS["a2c"], "pg"),
}
EXTRA_SEEDS = [2, 3]


def load_record(algorithm, name):
    return json.loads((LOGS_DIR / algorithm / f"{name}.json").read_text())


def select_best(algorithm):
    _, experiments, _ = ALGORITHMS[algorithm]
    records = [load_record(algorithm, name) for name, _ in experiments]
    return max(records, key=lambda r: r["best_validation_return"]), experiments


def train_seed_variant(job, episodes):
    algorithm, name, overrides = job
    if algorithm == "dqn":
        return dqn_training.train((name, overrides), episodes)
    return pg_training.train(algorithm, (name, overrides), episodes)


def seed_jobs(best):
    jobs = []
    for algorithm, (record, experiments) in best.items():
        overrides = dict(experiments)[record["name"]]
        for seed in EXTRA_SEEDS:
            jobs.append((algorithm, f"{record['name']}_seed{seed}", {**overrides, "seed": seed}))
    return jobs


def describe(summary):
    return {
        "return_mean": summary["return_mean"],
        "return_std": summary["return_std"],
        "on_time_rate": summary["on_time_rate_mean"],
        "unresolved": summary["unresolved_mean"],
        "mean_wait_minutes": summary["mean_wait_minutes_mean"],
        "mean_referral_minutes": summary["mean_referral_minutes_mean"],
        "ambulance_utilization": summary["ambulance_utilization_mean"],
        "total_km": summary["total_km_mean"],
        "invalid_actions": summary["invalid_actions_mean"],
        "hospital_admissions": summary["hospital_admissions"],
    }


def main():
    parser = argparse.ArgumentParser(description="Select the best model per algorithm and build the final comparison.")
    parser.add_argument("--skip-seeds", action="store_true", help="reuse existing seed-robustness runs")
    parser.add_argument("--episodes", type=int, default=TRAIN_EPISODES)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    best = {algorithm: select_best(algorithm) for algorithm in ALGORITHMS}
    jobs = seed_jobs(best)
    if not args.skip_seeds:
        run_many(functools.partial(train_seed_variant, episodes=args.episodes), jobs, args.workers)

    baselines = json.loads((LOGS_DIR / "baselines.json").read_text())
    comparison = {
        "selection_rule": "highest validation return (seeds 5000-5004); evaluation on held-out seeds 10000-10019",
        "baselines": {
            name: {scenario: describe(result["summary"]) for scenario, result in scenarios.items()}
            for name, scenarios in baselines.items()
        },
        "algorithms": {},
    }
    best_models = {}
    for algorithm, (record, _) in best.items():
        label, _, model_dir = ALGORITHMS[algorithm]
        seed_records = [record] + [load_record(algorithm, name) for a, name, _ in jobs if a == algorithm]
        seed_returns = [r["evaluation"]["standard"]["return_mean"] for r in seed_records]
        seed_on_time = [r["evaluation"]["standard"]["on_time_rate_mean"] for r in seed_records]
        comparison["algorithms"][algorithm] = {
            "label": label,
            "best_experiment": record["name"],
            "config": record["config"],
            "best_validation_return": record["best_validation_return"],
            "train_seconds": record["train_seconds"],
            "total_steps": record["total_steps"],
            "evaluation": {scenario: describe(summary) for scenario, summary in record["evaluation"].items()},
            "seed_robustness": {
                "seeds": [r["config"]["seed"] for r in seed_records],
                "eval_returns": seed_returns,
                "eval_return_mean": float(np.mean(seed_returns)),
                "eval_return_std": float(np.std(seed_returns)),
                "on_time_rates": seed_on_time,
            },
        }
        best_models[algorithm] = {
            "label": label,
            "experiment": record["name"],
            "path": f"models/{model_dir}/{record['name']}.pt",
            "eval_return": record["evaluation"]["standard"]["return_mean"],
        }
    overall = max(best_models, key=lambda a: comparison["algorithms"][a]["best_validation_return"])
    comparison["overall_best"] = overall
    (LOGS_DIR / "comparison.json").write_text(json.dumps(comparison, indent=2))
    (MODELS_DIR / "best_models.json").write_text(json.dumps({"overall_best": overall, "models": best_models}, indent=2))

    print(f"{'policy':<11} {'experiment':<30} {'val':>7} {'eval return':>17} {'on-time':>8} {'rainy':>8} {'3-seed mean ± std':>20}")
    for name in ("random", "nearest"):
        std, rainy = comparison["baselines"][name]["standard"], comparison["baselines"][name]["rainy_season"]
        print(f"{name:<11} {'(baseline)':<30} {'-':>7} {std['return_mean']:>8.1f} ± {std['return_std']:<6.1f} {std['on_time_rate']:>8.1%} {rainy['return_mean']:>8.1f} {'-':>20}")
    for algorithm, entry in comparison["algorithms"].items():
        std, rainy, seeds = entry["evaluation"]["standard"], entry["evaluation"]["rainy_season"], entry["seed_robustness"]
        print(
            f"{entry['label']:<11} {entry['best_experiment']:<30} {entry['best_validation_return']:>7.1f} "
            f"{std['return_mean']:>8.1f} ± {std['return_std']:<6.1f} {std['on_time_rate']:>8.1%} {rainy['return_mean']:>8.1f} "
            f"{seeds['eval_return_mean']:>11.1f} ± {seeds['eval_return_std']:<6.1f}"
        )
    print(f"Overall best (by validation): {comparison['algorithms'][overall]['label']} -> {best_models[overall]['path']}")
    print(f"Saved {LOGS_DIR / 'comparison.json'} and {MODELS_DIR / 'best_models.json'}")


if __name__ == "__main__":
    main()
