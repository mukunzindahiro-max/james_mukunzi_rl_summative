import json
import threading
import webbrowser
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from environment.custom_env import N_ACTIONS, OBS_SIZE, MaternalReferralEnv, _to_builtin, describe_action

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = Path(__file__).resolve().parent / "web"
ASSETS_DIR = ROOT / "assets"
BEST_MODELS_PATH = ROOT / "models" / "best_models.json"
COMPARISON_PATH = ROOT / "logs" / "comparison.json"
RL_POLICIES = ["dqn", "a2c", "ppo", "reinforce"]
POLICIES = RL_POLICIES + ["nearest", "random", "manual"]
BASELINE_LABELS = {
    "nearest": "Nearest-ambulance baseline",
    "random": "Random baseline",
    "manual": "Manual control",
}
DEFAULT_SEED = 10_000


def scenarios():
    from training.evaluation import SCENARIOS

    return SCENARIOS


def best_models():
    if not BEST_MODELS_PATH.exists():
        return {}
    return json.loads(BEST_MODELS_PATH.read_text())["models"]


class Agent:
    def __init__(self, key):
        if key not in POLICIES:
            raise ValueError(f"Unknown policy '{key}'. Choose from {POLICIES}")
        self.key = key
        self.score_type = None
        self.actor = None
        self.policy = None
        if key in RL_POLICIES:
            models = best_models()
            if key not in models:
                raise ValueError(f"No trained {key.upper()} model found. Run training and `python -m training.compare` first.")
            info = models[key]
            self.label = f"{info['label']} ({info['experiment']})"
            checkpoint = torch.load(ROOT / info["path"], map_location="cpu")
            if key == "dqn":
                from training.dqn_training import build_actor

                self.actor = build_actor(checkpoint["config"]["hidden"])
                self.actor.load_state_dict(checkpoint["state_dict"])
                self.score_type = "Q-value"
            else:
                from training.pg_training import build_actor

                self.actor = build_actor(checkpoint["config"]["hidden"])
                self.actor.load_state_dict(checkpoint["actor"])
                self.score_type = "probability"
            self.actor.eval()
        else:
            from training.evaluation import nearest_policy, random_policy

            self.label = BASELINE_LABELS[key]
            if key == "nearest":
                self.policy = nearest_policy()
            elif key == "random":
                self.policy = random_policy(0)

    def act(self, env, obs):
        if self.policy is not None:
            return self.policy(env, obs), []
        from training.common import obs_to_td

        td = obs_to_td(obs)
        with torch.no_grad():
            if self.key == "dqn":
                values = self.actor(td)["action_value"]
            else:
                values = self.actor.get_dist(td).probs
        values = values.numpy()
        valid = np.flatnonzero(obs["action_mask"])
        action = int(valid[np.argmax(values[valid])])
        scores = sorted(
            ({"action": int(a), "description": describe_action(a), "score": float(values[a])} for a in valid),
            key=lambda s: -s["score"],
        )[:5]
        return action, scores


def referral_text(view):
    if view is None:
        return "no pending referral"
    needs = [n for n, flag in (("surgery", view["needs_surgery"]), ("neonatal", view["needs_neonatal"])) if flag]
    return f"#{view['id']:<2} {view['priority_name']:<8} @ {view['origin_name']:<24} needs {'+'.join(needs) or 'basic':<16}"


class Session:
    def __init__(self, policy="dqn", scenario="standard", seed=DEFAULT_SEED, verbose=True):
        self.lock = threading.Lock()
        self.verbose = verbose
        self.policy, self.scenario, self.seed = policy, scenario, seed
        self.reset(policy, scenario, seed)

    def reset(self, policy=None, scenario=None, seed=None):
        policy = policy or self.policy
        scenario = scenario or self.scenario
        seed = self.seed if seed is None else int(seed)
        if scenario not in scenarios():
            raise ValueError(f"Unknown scenario '{scenario}'")
        self.agent = Agent(policy)
        self.policy, self.scenario, self.seed = policy, scenario, seed
        self.env = MaternalReferralEnv(**scenarios()[scenario])
        self.obs, _ = self.env.reset(seed=seed)
        self.steps = 0
        self.total_reward = 0.0
        self.done = False
        self.decision = None
        if self.verbose:
            print("=" * 110)
            print(f"New episode | policy: {self.agent.label} | scenario: {scenario} | seed: {seed}")
            print("=" * 110, flush=True)

    def step(self, action=None):
        if self.done:
            return
        scores = []
        if action is None:
            if self.policy == "manual":
                raise ValueError("Manual mode: choose an action")
            action, scores = self.agent.act(self.env, self.obs)
        if not 0 <= int(action) < N_ACTIONS:
            raise ValueError(f"Action must be in [0, {N_ACTIONS - 1}]")
        action = int(action)
        referral = self.env._current_referral()
        view = None if referral is None else self.env._referral_view(referral)
        decision_time = self.env.t
        clock = self.env._clock()
        self.obs, reward, terminated, truncated, info = self.env.step(action)
        self.steps += 1
        self.total_reward += reward
        self.done = bool(terminated or truncated)
        self.decision = {
            "step": self.steps,
            "time": decision_time,
            "clock": clock,
            "referral": view,
            "action": action,
            "description": describe_action(action),
            "outcome": info["outcome"],
            "reward": reward,
            "reward_parts": info["reward_parts"],
            "scores": scores,
            "score_type": self.agent.score_type,
        }
        if self.verbose:
            self._print_step()

    def _print_step(self):
        d = self.decision
        m = self.env.metrics()
        print(
            f"[{self.policy.upper():>9}] step {d['step']:>3} | {d['clock']} | {referral_text(d['referral'])} | "
            f"{d['description']:<52} | {d['outcome']:<9} | reward {d['reward']:+7.1f} | total {self.total_reward:8.1f} | "
            f"done {m['completed']:>2}/50 on-time {m['completed_on_time']:>2}",
            flush=True,
        )
        if self.done:
            print("-" * 110)
            print(
                f"Episode finished: return {self.total_reward:.1f} | completed {m['completed']} | on time "
                f"{m['completed_on_time']} ({m['on_time_rate']:.1%}) | unresolved {m['unresolved']} | "
                f"mean wait {m['mean_wait_minutes']:.1f} min | mean referral time {m['mean_referral_minutes']:.1f} min | "
                f"ambulance utilization {m['ambulance_utilization']:.1%} | km {m['total_km']:.0f}"
            )
            print(f"Admissions per hospital: {dict(zip(['District', 'Eastern Rural', 'Provincial'], m['hospital_admissions']))}")
            print("-" * 110, flush=True)

    def state(self):
        state = self.env.get_state()
        referral = self.env._current_referral()
        state.update(
            {
                "policy": self.policy,
                "policy_label": self.agent.label,
                "scenario": self.scenario,
                "seed": self.seed,
                "step": self.steps,
                "total_reward": self.total_reward,
                "done": self.done,
                "decision": self.decision,
                "current_referral_detail": None if referral is None else self.env._referral_view(referral),
                "valid_actions": [
                    {"action": int(a), "description": describe_action(a)} for a in np.flatnonzero(self.obs["action_mask"])
                ],
            }
        )
        return _to_builtin(state)


class ResetRequest(BaseModel):
    policy: str | None = None
    scenario: str | None = None
    seed: int | None = None


class StepRequest(BaseModel):
    action: int | None = None


class ActRequest(BaseModel):
    policy: str = "dqn"
    observation: list[float]
    action_mask: list[int]


_AGENTS = {}


def recommend(policy, observation, action_mask):
    if policy not in RL_POLICIES:
        raise ValueError(f"policy must be one of {RL_POLICIES}")
    if len(observation) != OBS_SIZE:
        raise ValueError(f"observation must have {OBS_SIZE} numbers (got {len(observation)})")
    if len(action_mask) != N_ACTIONS:
        raise ValueError(f"action_mask must have {N_ACTIONS} values (got {len(action_mask)})")
    mask = np.asarray(action_mask, dtype=np.int8)
    if not mask.any():
        raise ValueError("action_mask must allow at least one action")
    if policy not in _AGENTS:
        _AGENTS[policy] = Agent(policy)
    agent = _AGENTS[policy]
    obs = {"observation": np.asarray(observation, dtype=np.float32), "action_mask": mask}
    action, scores = agent.act(None, obs)
    return {
        "policy": policy,
        "policy_label": agent.label,
        "action": action,
        "description": describe_action(action),
        "score_type": agent.score_type,
        "scores": scores,
    }


def create_app(session):
    app = FastAPI(title="Maternal Emergency Referral RL Simulation")
    ASSETS_DIR.mkdir(exist_ok=True)
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")

    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/policies")
    def policies():
        models = best_models()
        return [
            {
                "key": key,
                "label": f"{models[key]['label']} ({models[key]['experiment']})" if key in models else BASELINE_LABELS.get(key, key.upper()),
                "available": key in models or key not in RL_POLICIES,
            }
            for key in POLICIES
        ]

    @app.get("/api/scenarios")
    def scenario_list():
        return list(scenarios())

    @app.get("/api/state")
    def get_state():
        with session.lock:
            return session.state()

    @app.post("/api/reset")
    def reset(request: ResetRequest):
        with session.lock:
            try:
                session.reset(request.policy, request.scenario, request.seed)
            except ValueError as err:
                raise HTTPException(status_code=400, detail=str(err))
            return session.state()

    @app.post("/api/step")
    def step(request: StepRequest):
        with session.lock:
            try:
                session.step(request.action)
            except ValueError as err:
                raise HTTPException(status_code=400, detail=str(err))
            return session.state()

    @app.post("/api/act")
    def act(request: ActRequest):
        try:
            return recommend(request.policy, request.observation, request.action_mask)
        except ValueError as err:
            raise HTTPException(status_code=422, detail=str(err))

    @app.get("/api/comparison")
    def comparison():
        return json.loads(COMPARISON_PATH.read_text()) if COMPARISON_PATH.exists() else {}

    @app.get("/api/figures")
    def figures():
        return sorted(p.name for p in ASSETS_DIR.glob("*.png"))

    return app


def serve(policy="dqn", scenario="standard", seed=DEFAULT_SEED, host="127.0.0.1", port=8000, open_browser=True):
    session = Session(policy, scenario, seed)
    app = create_app(session)
    url = f"http://{host}:{port}"
    print(f"3D simulation running at {url}  (Ctrl+C to stop)", flush=True)
    if open_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")


def run_terminal_episode(policy="dqn", scenario="standard", seed=DEFAULT_SEED, show_state=False):
    session = Session(policy, scenario, seed)
    while not session.done:
        session.step()
        if show_state:
            print(session.env._render_text(), flush=True)
    return session.env.metrics(), session.total_reward
