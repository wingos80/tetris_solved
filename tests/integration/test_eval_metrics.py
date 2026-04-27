"""Smoke tests for eval_metrics.py — ensures the eval functions run without
error on freshly-built (untrained) policies for every algo."""

import pytest
import torch

from eval_metrics import eval_afterstate_agent, eval_tianshou_agent
from eval import BUILDERS, _PATH_AWARE_BUILDERS


@pytest.mark.parametrize("algo", ["ppo", "rainbow", "sac", "dqn"])
def test_tianshou_eval_runs(tmp_path, algo):
    """Builds an untrained policy, saves weights, runs 1 episode of eval."""
    policy = BUILDERS[algo]()
    weights_path = tmp_path / "best.pth"
    torch.save(policy.state_dict(), weights_path)

    rewards, lines, placements = eval_tianshou_agent(str(weights_path), algo, episodes=1)
    assert len(rewards) == 1
    assert len(lines) == 1
    assert len(placements) == 1
    assert placements[0] >= 0
    assert lines[0] >= 0


def test_afterstate_eval_runs(tmp_path):
    """Builds an untrained afterstate net, saves weights + config, runs eval."""
    import json
    from rl_training.afterstate_dqn import VNet
    cfg = {"hidden": [32, 32]}
    net = VNet(cfg["hidden"], "cpu")
    weights_path = tmp_path / "best.pth"
    torch.save(net.state_dict(), weights_path)
    with open(tmp_path / "config.json", "w") as f:
        json.dump(cfg, f)

    rewards, lines, placements = eval_afterstate_agent(str(weights_path), episodes=2)
    assert len(rewards) == 2
    assert len(lines) == 2
    assert len(placements) == 2
    assert all(p >= 0 for p in placements)
