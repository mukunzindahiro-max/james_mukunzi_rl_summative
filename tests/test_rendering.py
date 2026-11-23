import json

import pytest

from environment.rendering import Session, best_models


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
