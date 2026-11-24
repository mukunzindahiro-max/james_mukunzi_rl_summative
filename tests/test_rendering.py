import json

import pytest

from environment.custom_env import N_ACTIONS, OBS_SIZE
from environment.rendering import Session, best_models, recommend


@pytest.mark.skipif("dqn" not in best_models(), reason="trained models not available")
def test_best_agent_session_runs_full_episode():
    session = Session("dqn", seed=10_000, verbose=False)
    while not session.done:
        session.step()
        assert session.decision["outcome"] in ("dispatch", "wait")
    metrics = session.env.metrics()
    assert metrics["completed"] + metrics["unresolved"] == 50
    json.dumps(session.state())


def test_manual_session_requires_an_action():
    session = Session("manual", seed=10_001, verbose=False)
    with pytest.raises(ValueError):
        session.step()
    action = session.state()["valid_actions"][-1]["action"]
    session.step(action)
    assert session.decision["action"] == action


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError):
        Session("not_a_policy", verbose=False)


@pytest.mark.skipif("dqn" not in best_models(), reason="trained models not available")
def test_recommend_matches_live_session_decision():
    session = Session("dqn", seed=10_002, verbose=False)
    obs = session.obs
    result = recommend("dqn", obs["observation"].tolist(), obs["action_mask"].tolist())
    session.step()
    assert result["action"] == session.decision["action"]
    assert obs["action_mask"][result["action"]] == 1
    json.dumps(result)


@pytest.mark.parametrize(
    "policy, observation, mask",
    [
        ("nearest", [0.0] * OBS_SIZE, [1] * N_ACTIONS),
        ("dqn", [0.0] * (OBS_SIZE - 1), [1] * N_ACTIONS),
        ("dqn", [0.0] * OBS_SIZE, [1] * (N_ACTIONS - 1)),
        ("dqn", [0.0] * OBS_SIZE, [0] * N_ACTIONS),
    ],
)
def test_recommend_rejects_bad_input(policy, observation, mask):
    with pytest.raises(ValueError):
        recommend(policy, observation, mask)
