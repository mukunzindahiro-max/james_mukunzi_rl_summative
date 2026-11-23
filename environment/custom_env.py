import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

MINUTES_PER_DAY = 1440
EPISODE_DAYS = 30
EPISODE_MINUTES = EPISODE_DAYS * MINUTES_PER_DAY
N_REFERRALS = 50
N_HEALTH_CENTRES = 5
N_HOSPITALS = 3
N_AMBULANCES = 5
N_ACTIONS = 1 + N_AMBULANCES * N_HOSPITALS
WAIT_ACTION = 0

NODES = [
    ("Rubona Health Centre", "health_centre", 8.0, 24.0),
    ("Kagarama Health Centre", "health_centre", 16.0, 4.0),
    ("Nyakabanda Health Centre", "health_centre", 28.0, 6.0),
    ("Mubuga Health Centre", "health_centre", 38.0, 12.0),
    ("Gikomero Health Centre", "health_centre", 24.0, 28.0),
    ("District Hospital", "hospital", 20.0, 15.0),
    ("Eastern Rural Hospital", "hospital", 34.0, 22.0),
    ("Provincial Referral Hospital", "hospital", 4.0, 8.0),
    ("Junction A", "junction", 12.0, 16.0),
    ("Junction B", "junction", 20.0, 9.0),
    ("Junction C", "junction", 28.0, 14.0),
    ("Junction D", "junction", 14.0, 24.0),
    ("Junction E", "junction", 30.0, 24.0),
    ("Junction F", "junction", 8.0, 12.0),
    ("Junction G", "junction", 36.0, 6.0),
    ("Junction H", "junction", 22.0, 22.0),
]
MAP_WIDTH_KM = 40.0
MAP_HEIGHT_KM = 30.0

EDGES = [
    (7, 13, "paved"),
    (13, 8, "paved"),
    (8, 5, "paved"),
    (5, 9, "paved"),
    (9, 1, "murram"),
    (13, 1, "rough"),
    (5, 10, "paved"),
    (10, 2, "murram"),
    (10, 3, "murram"),
    (14, 2, "rough"),
    (14, 3, "rough"),
    (10, 6, "paved"),
    (6, 12, "paved"),
    (12, 4, "murram"),
    (5, 15, "paved"),
    (15, 4, "murram"),
    (15, 11, "murram"),
    (11, 0, "murram"),
    (8, 11, "rough"),
    (0, 7, "rough"),
    (6, 3, "murram"),
    (12, 15, "murram"),
    (9, 14, "rough"),
]
ROAD_SPEED_KMH = {"paved": 55.0, "murram": 30.0, "rough": 18.0}
RAIN_SPEED_FACTOR = {"paved": 0.9, "murram": 0.6, "rough": 0.4}
NIGHT_SPEED_FACTOR = 0.8
ROAD_WINDING_FACTOR = 1.35

HEALTH_CENTRE_NODES = [0, 1, 2, 3, 4]
HEALTH_CENTRE_REFERRAL_SHARE = [0.25, 0.2, 0.2, 0.2, 0.15]
HOSPITALS = [
    {"name": "District Hospital", "node": 5, "capacity": 6, "surgery": True, "neonatal": True, "background_rate": 0.35},
    {"name": "Eastern Rural Hospital", "node": 6, "capacity": 4, "surgery": True, "neonatal": False, "background_rate": 0.3},
    {"name": "Provincial Referral Hospital", "node": 7, "capacity": 8, "surgery": True, "neonatal": True, "background_rate": 0.4},
]
AMBULANCE_BASES = [5, 7, 0, 6, 2]

PRIORITY_NAMES = {1: "critical", 2: "urgent", 3: "moderate"}
PRIORITY_PROBS = [0.3, 0.4, 0.3]
PRIORITY_WEIGHT = {1: 3.0, 2: 2.0, 3: 1.0}
TARGET_MINUTES = {1: 150.0, 2: 240.0, 3: 360.0}
EXCESSIVE_WAIT_MINUTES = {1: 30.0, 2: 60.0, 3: 120.0}
LENGTH_OF_STAY_DAYS = {1: (1.5, 3.0), 2: (1.0, 2.0), 3: (0.5, 1.5)}
MAX_WAIT_MINUTES = 480.0
LOAD_MINUTES = 15.0
HANDOVER_MINUTES = 20.0
COVERAGE_MINUTES = 45.0
WAIT_STEP_MINUTES = 30.0
UNNECESSARY_PICKUP_MARGIN_MINUTES = 45.0
REDUCED_TRAVEL_TOLERANCE = 1.1
FAILURE_UNRESOLVED_REFERRALS = 10
N_ARRIVAL_CLUSTERS = 14
CLUSTERED_ARRIVAL_SHARE = 0.6
CLUSTER_SPREAD_MINUTES = 240.0
LAST_ARRIVAL_MARGIN_MINUTES = 720.0
GENERAL_DUTY_PER_DAY = 2.0
GENERAL_DUTY_MINUTES = (60.0, 180.0)
RAIN_PROB = 0.35
SURGERY_OUTAGE_PROB = 0.08
NEONATAL_OUTAGE_PROB = 0.1
TIME_NORM_MINUTES = 240.0
DISTANCE_NORM_KM = 60.0
BUDGET_NORM_KM = 4000.0
PENDING_NORM = 10.0

DEFAULT_REWARDS = {
    "dispatch_appropriate": 10.0,
    "completed_on_time": 20.0,
    "reduced_travel": 5.0,
    "distribution_improved": 5.0,
    "excessive_wait": -10.0,
    "unnecessary_dispatch": -15.0,
    "unavailable_hospital": -30.0,
    "invalid_ambulance": -30.0,
    "prohibited_action": -100.0,
    "time_penalty_per_minute": -0.01,
    "unresolved_referral": -50.0,
}

REFERRAL_FEATURES = 12
AMBULANCE_FEATURES = 6
HOSPITAL_FEATURES = 6
GLOBAL_FEATURES = 12
OBS_SIZE = REFERRAL_FEATURES + N_AMBULANCES * AMBULANCE_FEATURES + N_HOSPITALS * HOSPITAL_FEATURES + GLOBAL_FEATURES


def decode_action(action):
    if action == WAIT_ACTION:
        return None, None
    return divmod(action - 1, N_HOSPITALS)


def encode_action(ambulance, hospital):
    return 1 + ambulance * N_HOSPITALS + hospital


def describe_action(action):
    ambulance, hospital = decode_action(int(action))
    if ambulance is None:
        return "Wait"
    return f"Dispatch Ambulance {ambulance + 1} -> {HOSPITALS[hospital]['name']}"


def _to_builtin(value):
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _build_road_tables():
    coords = np.array([[x, y] for _, _, x, y in NODES], dtype=np.float64)
    n = len(NODES)
    tables = {}
    for rain in (False, True):
        for night in (False, True):
            minutes = np.full((n, n), np.inf)
            km = np.full((n, n), np.inf)
            nxt = -np.ones((n, n), dtype=np.int64)
            for i in range(n):
                minutes[i, i] = 0.0
                km[i, i] = 0.0
                nxt[i, i] = i
            for u, v, road in EDGES:
                length = float(np.linalg.norm(coords[u] - coords[v])) * ROAD_WINDING_FACTOR
                speed = ROAD_SPEED_KMH[road]
                if rain:
                    speed *= RAIN_SPEED_FACTOR[road]
                if night:
                    speed *= NIGHT_SPEED_FACTOR
                for a, b in ((u, v), (v, u)):
                    minutes[a, b] = length / speed * 60.0
                    km[a, b] = length
                    nxt[a, b] = b
            for k in range(n):
                for i in range(n):
                    for j in range(n):
                        candidate = minutes[i, k] + minutes[k, j]
                        if candidate < minutes[i, j]:
                            minutes[i, j] = candidate
                            km[i, j] = km[i, k] + km[k, j]
                            nxt[i, j] = nxt[i, k]
            tables[(rain, night)] = (minutes, km, nxt)
    return coords, tables


NODE_COORDS, ROAD_TABLES = _build_road_tables()


class MaternalReferralEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "human"], "render_fps": 4}

    def __init__(
        self,
        render_mode=None,
        reward_config=None,
        max_steps=3000,
        rain_prob=RAIN_PROB,
        general_duty_per_day=GENERAL_DUTY_PER_DAY,
    ):
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode {render_mode}")
        self.render_mode = render_mode
        self.rewards = {**DEFAULT_REWARDS, **(reward_config or {})}
        self.max_steps = max_steps
        self.rain_prob = rain_prob
        self.general_duty_per_day = general_duty_per_day
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(-1.0, 1.0, shape=(OBS_SIZE,), dtype=np.float32),
                "action_mask": spaces.MultiBinary(N_ACTIONS),
            }
        )
        self.t = 0.0
        self.steps = 0
        self.referrals = []
        self.ambulances = []
        self.admissions = [[] for _ in range(N_HOSPITALS)]
        self.events = []
        self.last_decision = None
        self._reward_parts = {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        self.t = 0.0
        self.steps = 0
        self.total_km = 0.0
        self.rain = rng.random(EPISODE_DAYS) < self.rain_prob
        self.surgery_ready = np.zeros((EPISODE_DAYS, N_HOSPITALS), dtype=bool)
        self.neonatal_ready = np.zeros((EPISODE_DAYS, N_HOSPITALS), dtype=bool)
        self.background = np.zeros((EPISODE_DAYS, N_HOSPITALS), dtype=np.int64)
        for h, spec in enumerate(HOSPITALS):
            self.surgery_ready[:, h] = spec["surgery"] & (rng.random(EPISODE_DAYS) >= SURGERY_OUTAGE_PROB)
            self.neonatal_ready[:, h] = spec["neonatal"] & (rng.random(EPISODE_DAYS) >= NEONATAL_OUTAGE_PROB)
            self.background[:, h] = rng.binomial(spec["capacity"], spec["background_rate"], size=EPISODE_DAYS)
        self.referrals = self._generate_referrals(rng)
        self.ambulances = [
            {
                "id": i,
                "base": base,
                "busy_until": 0.0,
                "task": None,
                "waypoints": [(0.0, base)],
                "referral": None,
                "trips": [],
                "km": 0.0,
                "duties": self._generate_duties(rng),
            }
            for i, base in enumerate(AMBULANCE_BASES)
        ]
        self.admissions = [[] for _ in range(N_HOSPITALS)]
        self.events = []
        self.last_decision = None
        self._reward_parts = {}
        self._log("Simulation started: 5 ambulances at base, 3 hospitals ready")
        self._advance_until_decision()
        self._reward_parts = {}
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), self._get_info("reset")

    def step(self, action):
        action = int(action)
        self.steps += 1
        self._reward_parts = {}
        referral = self._current_referral()
        outcome = "wait"
        if action != WAIT_ACTION and referral is not None:
            ambulance, hospital = decode_action(action)
            if not self._hospital_suitable(hospital, referral):
                outcome = "prohibited_action"
                self._add_reward("prohibited_action", self.rewards["prohibited_action"])
            elif not self._ambulance_idle(ambulance):
                outcome = "invalid_ambulance"
                self._add_reward("invalid_ambulance", self.rewards["invalid_ambulance"])
            elif self._free_beds(hospital) <= 0:
                outcome = "unavailable_hospital"
                self._add_reward("unavailable_hospital", self.rewards["unavailable_hospital"])
            else:
                outcome = "dispatch"
                self._dispatch(referral, ambulance, hospital)
        if outcome != "dispatch":
            self._advance(self._next_wait_time())
        self.last_decision = {
            "action": action,
            "description": describe_action(action),
            "outcome": outcome,
            "referral": None if referral is None else referral["id"],
            "time": self.t,
        }
        if outcome != "dispatch" and referral is not None:
            self._log(f"Decision: {describe_action(action)} for referral #{referral['id']} ({outcome})")
        self._advance_until_decision()
        terminated, truncated = self._check_end()
        reward = float(sum(self._reward_parts.values()))
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, truncated, self._get_info(outcome)

    def action_mask(self):
        mask = np.zeros(N_ACTIONS, dtype=np.int8)
        mask[WAIT_ACTION] = 1
        referral = self._current_referral()
        if referral is None:
            return mask
        for a in range(N_AMBULANCES):
            if not self._ambulance_idle(a):
                continue
            for h in range(N_HOSPITALS):
                if self._hospital_suitable(h, referral) and self._free_beds(h) > 0:
                    mask[encode_action(a, h)] = 1
        return mask

    def metrics(self):
        completed = [r for r in self.referrals if r["status"] == "completed"]
        dispatched = [r for r in self.referrals if r["dispatch_time"] is not None]
        unresolved = sum(r["status"] == "unresolved" for r in self.referrals)
        elapsed = max(self.t, 1.0)
        busy = sum(min(end, self.t) - start for amb in self.ambulances for start, end in amb["trips"] if start < self.t)
        return {
            "time_minutes": self.t,
            "referrals_arrived": sum(r["status"] != "upcoming" for r in self.referrals),
            "completed": len(completed),
            "completed_on_time": sum(bool(r["on_time"]) for r in completed),
            "on_time_rate": sum(bool(r["on_time"]) for r in completed) / max(len(completed), 1),
            "unresolved": unresolved,
            "pending": len(self._pending()),
            "mean_wait_minutes": float(np.mean([r["dispatch_time"] - r["arrival"] for r in dispatched])) if dispatched else 0.0,
            "mean_referral_minutes": float(np.mean([r["hospital_arrival"] - r["arrival"] for r in completed])) if completed else 0.0,
            "ambulance_utilization": busy / (N_AMBULANCES * elapsed),
            "total_km": self.total_km,
            "coverage": self._coverage(),
            "hospital_admissions": [sum(r["hospital"] == h for r in dispatched) for h in range(N_HOSPITALS)],
        }

    def render(self):
        text = self._render_text()
        if self.render_mode == "human":
            print(text)
            return None
        return text

    def get_state(self):
        current = self._current_referral()
        return _to_builtin({
            "time_minutes": self.t,
            "day": self._day() + 1,
            "clock": self._clock(),
            "raining": bool(self.rain[self._day()]),
            "night": self._is_night(),
            "nodes": [{"id": i, "name": n, "type": k, "x": x, "y": y} for i, (n, k, x, y) in enumerate(NODES)],
            "edges": [{"from": u, "to": v, "road": road} for u, v, road in EDGES],
            "ambulances": [
                {
                    "id": a["id"],
                    "base": a["base"],
                    "status": "available" if self._ambulance_idle(a["id"]) else a["task"],
                    "position": list(self._ambulance_position(a["id"])),
                    "busy_until": a["busy_until"],
                    "waypoints": [[t, n] for t, n in a["waypoints"]],
                    "referral": a["referral"],
                }
                for a in self.ambulances
            ],
            "hospitals": [
                {
                    "id": h,
                    "name": spec["name"],
                    "node": spec["node"],
                    "capacity": spec["capacity"],
                    "occupancy": self._occupancy(h),
                    "free_beds": self._free_beds(h),
                    "surgery_ready": bool(self.surgery_ready[self._day(), h]),
                    "neonatal_ready": bool(self.neonatal_ready[self._day(), h]),
                }
                for h, spec in enumerate(HOSPITALS)
            ],
            "referrals": [self._referral_view(r) for r in self.referrals if r["status"] != "upcoming"],
            "current_referral": None if current is None else current["id"],
            "action_mask": self.action_mask().tolist(),
            "last_decision": self.last_decision,
            "metrics": self.metrics(),
            "events": self.events[-40:],
        })

    def _generate_referrals(self, rng):
        latest = EPISODE_MINUTES - LAST_ARRIVAL_MARGIN_MINUTES
        centres = rng.uniform(0.0, latest, size=N_ARRIVAL_CLUSTERS)
        times = []
        for _ in range(N_REFERRALS):
            if rng.random() < CLUSTERED_ARRIVAL_SHARE:
                t = centres[rng.integers(N_ARRIVAL_CLUSTERS)] + rng.uniform(0.0, CLUSTER_SPREAD_MINUTES)
            else:
                t = rng.uniform(0.0, latest)
            times.append(float(min(t, latest)))
        times.sort()
        referrals = []
        for i, t in enumerate(times):
            priority = int(rng.choice([1, 2, 3], p=PRIORITY_PROBS))
            if priority == 1:
                surgery, neonatal = True, rng.random() < 0.5
            elif priority == 2:
                surgery, neonatal = rng.random() < 0.5, rng.random() < 0.25
            else:
                surgery, neonatal = False, rng.random() < 0.1
            referrals.append(
                {
                    "id": i,
                    "arrival": t,
                    "origin": int(rng.choice(N_HEALTH_CENTRES, p=HEALTH_CENTRE_REFERRAL_SHARE)),
                    "priority": priority,
                    "needs_surgery": bool(surgery),
                    "needs_neonatal": bool(neonatal),
                    "status": "upcoming",
                    "ambulance": None,
                    "hospital": None,
                    "dispatch_time": None,
                    "pickup_time": None,
                    "hospital_arrival": None,
                    "wait_flagged": False,
                    "on_time": None,
                }
            )
        return referrals

    def _generate_duties(self, rng):
        count = rng.poisson(self.general_duty_per_day * EPISODE_DAYS)
        starts = np.sort(rng.uniform(0.0, EPISODE_MINUTES, size=count))
        return [[float(s), float(s + rng.uniform(*GENERAL_DUTY_MINUTES))] for s in starts]

    def _start_duties(self, t_old, t_new):
        for amb in self.ambulances:
            while amb["duties"] and amb["duties"][0][0] <= t_new:
                start, end = amb["duties"].pop(0)
                if start <= t_old or amb["busy_until"] > start:
                    continue
                amb.update(busy_until=end, task="general_duty", referral=None, waypoints=[(start, amb["base"]), (end, amb["base"])])
                amb["trips"].append((start, end))
                self._log(f"Ambulance {amb['id'] + 1} called to general emergency duty until {self._clock(end)}", start)

    def _dispatch(self, referral, ambulance, hospital):
        minutes, km, _ = self._table()
        amb = self.ambulances[ambulance]
        base = amb["base"]
        origin = HEALTH_CENTRE_NODES[referral["origin"]]
        hospital_node = HOSPITALS[hospital]["node"]

        valid = [decode_action(a) for a in np.flatnonzero(self.action_mask()) if a != WAIT_ACTION]
        trip_km = {(a, h): km[self.ambulances[a]["base"], origin] + km[origin, HOSPITALS[h]["node"]] for a, h in valid}
        pickups = {a: minutes[self.ambulances[a]["base"], origin] for a, _ in valid}
        coverage_after = {a: self._coverage(exclude=a) for a in pickups}

        self._add_reward("dispatch_appropriate", self.rewards["dispatch_appropriate"])
        if trip_km[(ambulance, hospital)] <= REDUCED_TRAVEL_TOLERANCE * min(trip_km.values()) + 1e-6:
            self._add_reward("reduced_travel", self.rewards["reduced_travel"])
        if coverage_after[ambulance] >= max(coverage_after.values()) - 1e-9:
            self._add_reward("distribution_improved", self.rewards["distribution_improved"])
        if pickups[ambulance] > min(pickups.values()) + UNNECESSARY_PICKUP_MARGIN_MINUTES:
            self._add_reward("unnecessary_dispatch", self.rewards["unnecessary_dispatch"])

        t0 = self.t
        waypoints = self._route(base, origin, t0)
        waypoints.append((waypoints[-1][0] + LOAD_MINUTES, origin))
        pickup_time = waypoints[-2][0]
        waypoints += self._route(origin, hospital_node, waypoints[-1][0])[1:]
        hospital_arrival = waypoints[-1][0]
        waypoints.append((hospital_arrival + HANDOVER_MINUTES, hospital_node))
        waypoints += self._route(hospital_node, base, waypoints[-1][0])[1:]
        free_at = waypoints[-1][0]
        distance = km[base, origin] + km[origin, hospital_node] + km[hospital_node, base]

        low, high = LENGTH_OF_STAY_DAYS[referral["priority"]]
        self.admissions[hospital].append(hospital_arrival + self.np_random.uniform(low, high) * MINUTES_PER_DAY)
        amb.update(busy_until=free_at, task="referral", waypoints=waypoints, referral=referral["id"])
        amb["trips"].append((t0, free_at))
        amb["km"] += distance
        self.total_km += distance
        referral.update(
            status="in_transit",
            ambulance=ambulance,
            hospital=hospital,
            dispatch_time=t0,
            pickup_time=pickup_time,
            hospital_arrival=hospital_arrival,
        )
        self._log(
            f"Dispatch: Ambulance {ambulance + 1} -> referral #{referral['id']} "
            f"({PRIORITY_NAMES[referral['priority']]}) -> {HOSPITALS[hospital]['name']}, "
            f"ETA hospital {self._clock(hospital_arrival)}"
        )

    def _advance(self, t_new):
        t_new = min(t_new, EPISODE_MINUTES)
        t_old = self.t
        if t_new <= t_old:
            return
        self._start_duties(t_old, t_new)
        for r in self.referrals:
            if r["status"] in ("upcoming", "pending"):
                if r["arrival"] > t_new:
                    continue
                if r["status"] == "upcoming":
                    r["status"] = "pending"
                    self._log(
                        f"New referral #{r['id']}: {PRIORITY_NAMES[r['priority']]} at "
                        f"{NODES[HEALTH_CENTRE_NODES[r['origin']]][0]}",
                        r["arrival"],
                    )
                deadline = r["arrival"] + MAX_WAIT_MINUTES
                start, end = max(t_old, r["arrival"]), min(t_new, deadline)
                if end > start:
                    self._add_reward(
                        "time_penalty",
                        self.rewards["time_penalty_per_minute"] * PRIORITY_WEIGHT[r["priority"]] * (end - start),
                    )
                threshold = r["arrival"] + EXCESSIVE_WAIT_MINUTES[r["priority"]]
                if not r["wait_flagged"] and threshold <= t_new and threshold < deadline:
                    r["wait_flagged"] = True
                    self._add_reward("excessive_wait", self.rewards["excessive_wait"])
                if deadline <= t_new:
                    r["status"] = "unresolved"
                    self._add_reward("unresolved_referral", self.rewards["unresolved_referral"])
                    self._log(f"Referral #{r['id']} unresolved after {MAX_WAIT_MINUTES:.0f} min wait", deadline)
            elif r["status"] == "in_transit" and r["hospital_arrival"] <= t_new:
                r["status"] = "completed"
                r["on_time"] = r["hospital_arrival"] - r["arrival"] <= TARGET_MINUTES[r["priority"]]
                if r["on_time"]:
                    self._add_reward("completed_on_time", self.rewards["completed_on_time"])
                self._log(
                    f"Referral #{r['id']} arrived at {HOSPITALS[r['hospital']]['name']} "
                    f"({'on time' if r['on_time'] else 'late'})",
                    r["hospital_arrival"],
                )
        self.t = t_new

    def _advance_until_decision(self):
        while self.t < EPISODE_MINUTES and not self._pending():
            upcoming = [r["arrival"] for r in self.referrals if r["status"] == "upcoming"]
            if upcoming:
                self._advance(min(upcoming))
                continue
            in_transit = [r["hospital_arrival"] for r in self.referrals if r["status"] == "in_transit"]
            if in_transit:
                self._advance(max(in_transit))
            break

    def _next_wait_time(self):
        candidates = [r["arrival"] for r in self.referrals if r["status"] == "upcoming"]
        candidates += [a["busy_until"] for a in self.ambulances]
        candidates += [a["duties"][0][0] for a in self.ambulances if a["duties"]]
        candidates += [d for beds in self.admissions for d in beds]
        candidates.append((self._day() + 1) * MINUTES_PER_DAY)
        for r in self._pending():
            candidates.append(r["arrival"] + EXCESSIVE_WAIT_MINUTES[r["priority"]])
            candidates.append(r["arrival"] + MAX_WAIT_MINUTES)
        future = [c for c in candidates if c > self.t]
        return max(min(future + [self.t + WAIT_STEP_MINUTES]), self.t + 1.0)

    def _check_end(self):
        unresolved = sum(r["status"] == "unresolved" for r in self.referrals)
        if self.t >= EPISODE_MINUTES:
            for r in self.referrals:
                if r["status"] in ("upcoming", "pending"):
                    r["status"] = "unresolved"
                    self._add_reward("unresolved_referral", self.rewards["unresolved_referral"])
        all_resolved = all(r["status"] in ("completed", "unresolved") for r in self.referrals)
        terminated = all_resolved or unresolved >= FAILURE_UNRESOLVED_REFERRALS or self.t >= EPISODE_MINUTES
        truncated = not terminated and self.steps >= self.max_steps
        return terminated, truncated

    def _get_obs(self):
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        referral = self._current_referral()
        minutes, km, _ = self._table()
        origin = None
        if referral is not None:
            origin = HEALTH_CENTRE_NODES[referral["origin"]]
            obs[0] = 1.0
            obs[referral["priority"]] = 1.0
            obs[4 + referral["origin"]] = 1.0
            obs[9] = float(referral["needs_surgery"])
            obs[10] = float(referral["needs_neonatal"])
            obs[11] = min((self.t - referral["arrival"]) / MAX_WAIT_MINUTES, 1.0)
        i = REFERRAL_FEATURES
        for a, amb in enumerate(self.ambulances):
            idle = self._ambulance_idle(a)
            x, y = self._ambulance_position(a)
            remaining = max(amb["busy_until"] - self.t, 0.0)
            obs[i] = float(idle)
            obs[i + 1] = x / MAP_WIDTH_KM
            obs[i + 2] = y / MAP_HEIGHT_KM
            obs[i + 3] = min(remaining / TIME_NORM_MINUTES, 1.0)
            if origin is None:
                obs[i + 4] = 1.0
                obs[i + 5] = 1.0
            else:
                obs[i + 4] = min((remaining + minutes[amb["base"], origin]) / TIME_NORM_MINUTES, 1.0)
                obs[i + 5] = min(km[amb["base"], origin] / DISTANCE_NORM_KM, 1.0)
            i += AMBULANCE_FEATURES
        day = self._day()
        for h, spec in enumerate(HOSPITALS):
            free = self._free_beds(h)
            obs[i] = free / spec["capacity"]
            obs[i + 1] = min(self._occupancy(h) / spec["capacity"], 1.0)
            obs[i + 2] = float(self.surgery_ready[day, h])
            obs[i + 3] = float(self.neonatal_ready[day, h])
            if origin is None:
                obs[i + 4] = 1.0
            else:
                obs[i + 4] = min(minutes[origin, spec["node"]] / TIME_NORM_MINUTES, 1.0)
                obs[i + 5] = float(self._hospital_suitable(h, referral) and free > 0)
            i += HOSPITAL_FEATURES
        pending = self._pending()
        waits = [self.t - r["arrival"] for r in pending]
        phase = 2.0 * math.pi * (self.t % MINUTES_PER_DAY) / MINUTES_PER_DAY
        obs[i] = min(len(pending) / PENDING_NORM, 1.0)
        obs[i + 1] = min(float(np.mean(waits)) / MAX_WAIT_MINUTES, 1.0) if waits else 0.0
        obs[i + 2] = min(max(waits) / MAX_WAIT_MINUTES, 1.0) if waits else 0.0
        obs[i + 3] = math.sin(phase)
        obs[i + 4] = math.cos(phase)
        obs[i + 5] = min(self.t / EPISODE_MINUTES, 1.0)
        obs[i + 6] = float(self.rain[day])
        obs[i + 7] = min(self.total_km / BUDGET_NORM_KM, 1.0)
        obs[i + 8] = self._coverage()
        obs[i + 9] = sum(r["status"] == "upcoming" for r in self.referrals) / N_REFERRALS
        obs[i + 10] = sum(r["status"] == "completed" for r in self.referrals) / N_REFERRALS
        obs[i + 11] = min(sum(r["status"] == "unresolved" for r in self.referrals) / FAILURE_UNRESOLVED_REFERRALS, 1.0)
        return {"observation": obs, "action_mask": self.action_mask()}

    def _get_info(self, outcome):
        referral = self._current_referral()
        return {
            "outcome": outcome,
            "time_minutes": self.t,
            "current_referral": -1 if referral is None else referral["id"],
            "reward_parts": dict(self._reward_parts),
        }

    def _add_reward(self, key, value):
        self._reward_parts[key] = self._reward_parts.get(key, 0.0) + float(value)

    def _log(self, message, t=None):
        t = self.t if t is None else t
        self.events.append({"time": t, "clock": self._clock(t), "message": message})

    def _pending(self):
        return [r for r in self.referrals if r["status"] == "pending"]

    def _current_referral(self):
        pending = self._pending()
        if not pending:
            return None
        return min(pending, key=lambda r: (r["priority"], r["arrival"]))

    def _day(self, t=None):
        t = self.t if t is None else t
        return min(int(t // MINUTES_PER_DAY), EPISODE_DAYS - 1)

    def _is_night(self, t=None):
        t = self.t if t is None else t
        hour = (t % MINUTES_PER_DAY) / 60.0
        return hour >= 20.0 or hour < 6.0

    def _clock(self, t=None):
        t = self.t if t is None else t
        minute = int(t % MINUTES_PER_DAY)
        return f"Day {self._day(t) + 1} {minute // 60:02d}:{minute % 60:02d}"

    def _table(self, t=None):
        return ROAD_TABLES[(bool(self.rain[self._day(t)]), self._is_night(t))]

    def _route(self, start, end, t_start):
        minutes, _, nxt = self._table(t_start)
        waypoints = [(t_start, start)]
        node, t = start, t_start
        while node != end:
            following = int(nxt[node, end])
            t += minutes[node, following]
            waypoints.append((t, following))
            node = following
        return waypoints

    def _ambulance_idle(self, ambulance):
        return self.ambulances[ambulance]["busy_until"] <= self.t

    def _ambulance_position(self, ambulance):
        waypoints = self.ambulances[ambulance]["waypoints"]
        if self.t >= waypoints[-1][0]:
            return tuple(NODE_COORDS[waypoints[-1][1]])
        for (t0, n0), (t1, n1) in zip(waypoints, waypoints[1:]):
            if t0 <= self.t < t1:
                frac = (self.t - t0) / (t1 - t0) if t1 > t0 else 1.0
                return tuple(NODE_COORDS[n0] + frac * (NODE_COORDS[n1] - NODE_COORDS[n0]))
        return tuple(NODE_COORDS[waypoints[0][1]])

    def _occupancy(self, hospital):
        active = sum(discharge > self.t for discharge in self.admissions[hospital])
        return int(self.background[self._day(), hospital]) + active

    def _free_beds(self, hospital):
        return max(HOSPITALS[hospital]["capacity"] - self._occupancy(hospital), 0)

    def _hospital_suitable(self, hospital, referral):
        day = self._day()
        if referral["needs_surgery"] and not self.surgery_ready[day, hospital]:
            return False
        if referral["needs_neonatal"] and not self.neonatal_ready[day, hospital]:
            return False
        return True

    def _coverage(self, exclude=None):
        minutes, _, _ = self._table()
        idle_bases = [a["base"] for a in self.ambulances if a["id"] != exclude and self._ambulance_idle(a["id"])]
        covered = sum(any(minutes[b, hc] <= COVERAGE_MINUTES for b in idle_bases) for hc in HEALTH_CENTRE_NODES)
        return covered / N_HEALTH_CENTRES

    def _referral_view(self, r):
        return {
            "id": r["id"],
            "priority": r["priority"],
            "priority_name": PRIORITY_NAMES[r["priority"]],
            "origin": HEALTH_CENTRE_NODES[r["origin"]],
            "origin_name": NODES[HEALTH_CENTRE_NODES[r["origin"]]][0],
            "needs_surgery": r["needs_surgery"],
            "needs_neonatal": r["needs_neonatal"],
            "status": r["status"],
            "arrival": r["arrival"],
            "waited_minutes": (r["dispatch_time"] if r["dispatch_time"] is not None else self.t) - r["arrival"],
            "ambulance": r["ambulance"],
            "hospital": r["hospital"],
            "dispatch_time": r["dispatch_time"],
            "pickup_time": r["pickup_time"],
            "hospital_arrival": r["hospital_arrival"],
            "on_time": r["on_time"],
        }

    def _render_text(self):
        day = self._day()
        lines = [
            f"=== {self._clock()} | {'rain' if self.rain[day] else 'dry'} | {'night' if self._is_night() else 'day'} ===",
        ]
        if self.last_decision is not None:
            lines.append(f"Last decision: {self.last_decision['description']} ({self.last_decision['outcome']})")
        referral = self._current_referral()
        if referral is not None:
            needs = [n for n, f in (("surgery", referral["needs_surgery"]), ("neonatal", referral["needs_neonatal"])) if f]
            lines.append(
                f"Current referral #{referral['id']}: {PRIORITY_NAMES[referral['priority']]} at "
                f"{NODES[HEALTH_CENTRE_NODES[referral['origin']]][0]}, needs {', '.join(needs) or 'basic care'}, "
                f"waiting {self.t - referral['arrival']:.0f} min"
            )
        lines.append(f"Pending referrals: {len(self._pending())}")
        for a, amb in enumerate(self.ambulances):
            status = "available" if self._ambulance_idle(a) else f"{amb['task']} until {self._clock(amb['busy_until'])}"
            lines.append(f"  Ambulance {a + 1} (base {NODES[amb['base']][0]}): {status}")
        for h, spec in enumerate(HOSPITALS):
            lines.append(
                f"  {spec['name']}: {self._free_beds(h)}/{spec['capacity']} beds free, "
                f"surgery {'ready' if self.surgery_ready[day, h] else 'down'}, "
                f"neonatal {'ready' if self.neonatal_ready[day, h] else 'down'}"
            )
        m = self.metrics()
        lines.append(
            f"Completed {m['completed']} (on time {m['completed_on_time']}), unresolved {m['unresolved']}, "
            f"utilization {m['ambulance_utilization']:.1%}, km {m['total_km']:.0f}"
        )
        return "\n".join(lines)
