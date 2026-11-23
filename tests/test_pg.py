import torch
from tensordict import TensorDict

from environment.custom_env import N_ACTIONS, OBS_SIZE
from training.pg_training import build_actor, discounted_returns


def masked_batch(size=64):
    mask = torch.zeros(size, N_ACTIONS, dtype=torch.bool)
    mask[:, 0] = True
    mask[::2, 7] = True
    return TensorDict({"observation": torch.randn(size, OBS_SIZE), "action_mask": mask}, [size])


def test_policy_samples_and_greedy_actions_respect_mask():
    torch.manual_seed(0)
    actor = build_actor([32])
    td = masked_batch()
    sampled = actor(td.clone())["action"]
    greedy = actor.get_dist(td.clone()).probs.argmax(-1)
    for actions in (sampled, greedy):
        assert td["action_mask"].gather(-1, actions.unsqueeze(-1)).all()


def test_discounted_returns_reset_at_episode_boundary():
    batch = TensorDict(
        {
            "next": TensorDict(
                {
                    "reward": torch.tensor([[1.0], [1.0], [2.0], [3.0]]),
                    "done": torch.tensor([[False], [True], [False], [True]]),
                },
                [4],
            )
        },
        [4],
    )
    returns = discounted_returns(batch, 0.5)
    assert torch.allclose(returns, torch.tensor([1.5, 1.0, 3.5, 3.0]))
