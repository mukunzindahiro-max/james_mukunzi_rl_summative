import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from environment.custom_env import (
    N_ACTIONS,
    OBS_SIZE,
    WAIT_ACTION,
    MaternalReferralEnv,
    decode_action,
    encode_action,
)


def run_random_episode(seed):
    env = MaternalReferralEnv()
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    total, steps, done = 0.0, 0, False
    while not done:
        action = int(rng.choice(np.flatnonzero(obs["action_mask"])))
        obs, reward, terminated, truncated, _ = env.step(action)
        assert env.observation_space.contains(obs)
        total += reward
        steps += 1
        done = terminated or truncated
    return env, total, steps


def test_gymnasium_api_compliance():
    check_env(MaternalReferralEnv(), skip_render_check=True)


def test_spaces_and_reset():
    env = MaternalReferralEnv()
    obs, info = env.reset(seed=0)
    assert env.action_space.n == N_ACTIONS == 16
    assert obs["observation"].shape == (OBS_SIZE,)
    assert env.observation_space.contains(obs)
    assert obs["action_mask"][WAIT_ACTION] == 1
    assert info["current_referral"] >= 0
    assert all(env._ambulance_idle(a) or env.ambulances[a]["task"] == "general_duty" for a in range(5))


def test_action_encoding_roundtrip():
    for action in range(1, N_ACTIONS):
        ambulance, hospital = decode_action(action)
        assert encode_action(ambulance, hospital) == action
    assert decode_action(WAIT_ACTION) == (None, None)


def test_mask_only_allows_feasible_dispatches():
    env = MaternalReferralEnv()
    obs, _ = env.reset(seed=1)
    referral = env._current_referral()
    for action in np.flatnonzero(obs["action_mask"]):
        if action == WAIT_ACTION:
            continue
        ambulance, hospital = decode_action(action)
        assert env._ambulance_idle(ambulance)
        assert env._hospital_suitable(hospital, referral)
        assert env._free_beds(hospital) > 0


def test_dispatch_changes_persistent_state():
    env = MaternalReferralEnv()
    obs, _ = env.reset(seed=2)
    action = int(next(a for a in np.flatnonzero(obs["action_mask"]) if a != WAIT_ACTION))
    ambulance, hospital = decode_action(action)
    referral_id = env._current_referral()["id"]
    dispatch_time = env.t
    admissions_before = len(env.admissions[hospital])
    _, _, _, _, info = env.step(action)
    assert info["outcome"] == "dispatch"
    assert info["reward_parts"]["dispatch_appropriate"] == 10.0
    assert len(env.admissions[hospital]) == admissions_before + 1
    assert env.referrals[referral_id]["ambulance"] == ambulance
    trip = next(t for t in env.ambulances[ambulance]["trips"] if t[0] == dispatch_time)
    assert trip[1] > dispatch_time


def test_invalid_ambulance_is_penalised_and_not_executed():
    env = MaternalReferralEnv()
    env.reset(seed=3)
    env.ambulances[0]["busy_until"] = 1e9
    referral_id = env._current_referral()["id"]
    hospital = next(h for h in range(3) if env._hospital_suitable(h, env._current_referral()))
    _, _, _, _, info = env.step(encode_action(0, hospital))
    assert info["outcome"] == "invalid_ambulance"
    assert info["reward_parts"]["invalid_ambulance"] == -30.0
    assert env.referrals[referral_id]["dispatch_time"] is None


def test_clinically_prohibited_action_is_blocked():
    env = MaternalReferralEnv()
    env.reset(seed=4)
    referral = env._current_referral()
    referral["needs_neonatal"] = True
    ambulance = next(a for a in range(5) if env._ambulance_idle(a))
    _, _, _, _, info = env.step(encode_action(ambulance, 1))
    assert info["outcome"] == "prohibited_action"
    assert info["reward_parts"]["prohibited_action"] == -100.0
    assert referral["dispatch_time"] is None


def test_full_hospital_is_unavailable():
    env = MaternalReferralEnv()
    env.reset(seed=5)
    referral = env._current_referral()
    referral["needs_surgery"] = False
    referral["needs_neonatal"] = False
    env.admissions[0] = [1e9] * 20
    ambulance = next(a for a in range(5) if env._ambulance_idle(a))
    _, _, _, _, info = env.step(encode_action(ambulance, 0))
    assert info["outcome"] == "unavailable_hospital"
    assert info["reward_parts"]["unavailable_hospital"] == -30.0


@pytest.mark.parametrize("seed", [0, 7, 21])
def test_episode_terminates_and_resolves_all_referrals(seed):
    env, _, steps = run_random_episode(seed)
    statuses = {r["status"] for r in env.referrals}
    assert statuses <= {"completed", "unresolved"}
    assert steps < env.max_steps
    metrics = env.metrics()
    assert metrics["completed"] + metrics["unresolved"] == 50
    assert 0.0 <= metrics["ambulance_utilization"] <= 1.0


def test_seeded_episodes_are_reproducible():
    _, total_a, steps_a = run_random_episode(11)
    _, total_b, steps_b = run_random_episode(11)
    assert total_a == pytest.approx(total_b)
    assert steps_a == steps_b


def test_state_snapshot_is_serialisable():
    import json

    env = MaternalReferralEnv()
    obs, _ = env.reset(seed=6)
    env.step(int(np.flatnonzero(obs["action_mask"])[-1]))
    state = env.get_state()
    json.dumps(state)
    assert len(state["ambulances"]) == 5
    assert len(state["hospitals"]) == 3
    assert len(state["nodes"]) == 16
