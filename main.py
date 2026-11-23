import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
COMMANDS = {
    "simulate": "Launch the Three.js 3D simulation with an agent (default)",
    "terminal": "Run one episode in the terminal only, with verbose decisions",
    "baselines": "Evaluate the random and nearest-ambulance baselines",
    "train-dqn": "Train all DQN experiments (TorchRL)",
    "train-pg": "Train all REINFORCE, PPO and A2C experiments (TorchRL)",
    "compare": "Select the best model per algorithm and build the final comparison",
}


def best_policy():
    path = ROOT / "models" / "best_models.json"
    return json.loads(path.read_text())["overall_best"] if path.exists() else "nearest"


def forward(module_main, rest):
    sys.argv = [sys.argv[0], *rest]
    module_main()


def main():
    parser = argparse.ArgumentParser(
        description="Maternal emergency referral & ambulance dispatch with reinforcement learning.",
        epilog="\n".join(f"  {name:<10} {text}" for name, text in COMMANDS.items()),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", nargs="?", default="simulate", choices=list(COMMANDS))
    parser.add_argument("--policy", default=None, help="dqn, a2c, ppo, reinforce, nearest, random or manual")
    parser.add_argument("--scenario", default="standard", choices=["standard", "rainy_season"])
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--show-state", action="store_true", help="terminal mode: print the full state after each step")
    args, rest = parser.parse_known_args()
    policy = args.policy or best_policy()

    if args.command == "simulate":
        from environment.rendering import serve

        serve(policy, args.scenario, args.seed, port=args.port, open_browser=not args.no_browser)
    elif args.command == "terminal":
        from environment.rendering import run_terminal_episode

        run_terminal_episode(policy, args.scenario, args.seed, show_state=args.show_state)
    elif args.command == "baselines":
        from training.evaluation import run_baselines

        run_baselines()
    elif args.command == "train-dqn":
        from training.dqn_training import main as dqn_main

        forward(dqn_main, rest)
    elif args.command == "train-pg":
        from training.pg_training import main as pg_main

        forward(pg_main, rest)
    elif args.command == "compare":
        from training.compare import main as compare_main

        forward(compare_main, rest)


if __name__ == "__main__":
    main()
