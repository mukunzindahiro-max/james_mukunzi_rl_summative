import torch
from tensordict import TensorDict

from environment.custom_env import N_ACTIONS, OBS_SIZE
from training.dqn_training import build_actor


def test_q_actor_respects_action_mask():
    torch.manual_seed(0)
    actor = build_actor([32])
    mask = torch.zeros(64, N_ACTIONS, dtype=torch.bool)
    mask[:, 0] = True
    mask[::2, 5] = True
    td = TensorDict({"observation": torch.randn(64, OBS_SIZE), "action_mask": mask}, [64])
    actions = actor(td)["action"]
    assert mask.gather(-1, actions.unsqueeze(-1)).all()
