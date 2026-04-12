"""Integration tests for eval.py — policy building, action selection, save/load.

Why each test exists:

- test_get_action_*: Every algo family has a different code path in get_action()
  (DQN-family uses policy(batch), SAC/PPO use policy.actor). Without these tests
  the DQN/QR-DQN/IQN .actor AttributeError crash went undetected.

- test_action_respects_mask: Ensures get_action never picks an invalid action.
  Mask bugs (wrong dtype, wrong dims) silently waste moves on illegal placements.

- test_build_*: Builder functions wire up specific architectures. Mismatches between
  builder and training code cause cryptic RuntimeErrors at eval time only.

- test_save_load_roundtrip: Catches state_dict key mismatches when network wrappers
  (MaskedSACActor, ObsExtractCritic, PlasticMLP) are refactored.
"""

import pytest
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "rl_training"))

from tetris.env import TetrisEnv

OBS_N, ACT_N = 78, 40
TINY = [32]  # small hidden sizes for fast tests


def _get_obs():
    """Get a real observation from the environment."""
    env = TetrisEnv()
    obs, _ = env.reset()
    return obs


# ── Policy builders (tiny networks for speed) ──────────────────────


def _build_ppo():
    from tianshou.policy import PPOPolicy
    from tianshou.utils.net.common import Net
    from tianshou.utils.net.discrete import Actor, Critic
    from torch.distributions import Categorical
    from ppo import MaskedActor, ObsExtractCritic

    net_a = Net(state_shape=OBS_N, hidden_sizes=TINY, device="cpu")
    net_c = Net(state_shape=OBS_N, hidden_sizes=TINY, device="cpu")
    raw_actor = Actor(net_a, ACT_N, device="cpu", softmax_output=False)
    raw_critic = Critic(net_c, device="cpu")
    actor = MaskedActor(raw_actor)
    critic = ObsExtractCritic(raw_critic)
    return PPOPolicy(
        actor, critic,
        torch.optim.Adam(list(actor.parameters()) + list(critic.parameters())),
        dist_fn=lambda logits: Categorical(logits=logits),
        action_space=TetrisEnv().action_space,
    )


def _build_rainbow():
    from tianshou.policy import RainbowPolicy
    from rainbow import RainbowNet

    net = RainbowNet(OBS_N, ACT_N, 32, 21, "cpu")
    return RainbowPolicy(
        net, torch.optim.Adam(net.parameters()),
        num_atoms=21, v_min=-10.0, v_max=10.0,
        estimation_step=3, target_update_freq=100,
    )


def _build_sac():
    from tianshou.policy import DiscreteSACPolicy
    from tianshou.utils.net.common import Net
    from sac import MaskedSACActor, ObsExtractCritic

    net_a = Net(state_shape=OBS_N, hidden_sizes=TINY, device="cpu")
    net_c1 = Net(state_shape=OBS_N, hidden_sizes=TINY, device="cpu")
    net_c2 = Net(state_shape=OBS_N, hidden_sizes=TINY, device="cpu")
    actor = MaskedSACActor(net_a, ACT_N, "cpu")
    critic1 = ObsExtractCritic(net_c1, ACT_N, "cpu")
    critic2 = ObsExtractCritic(net_c2, ACT_N, "cpu")
    return DiscreteSACPolicy(
        actor, torch.optim.Adam(actor.parameters()),
        critic1, torch.optim.Adam(critic1.parameters()),
        critic2, torch.optim.Adam(critic2.parameters()),
        tau=0.005, gamma=0.99, alpha=0.2, estimation_step=3,
    )


def _build_sac_v2():
    from tianshou.policy import DiscreteSACPolicy
    from sac_v2 import MaskedSACActor, ObsExtractCritic, PlasticMLP

    # Build PlasticMLP directly on CPU (avoid _make_net which uses module-level DEVICE)
    net_a = PlasticMLP(OBS_N, TINY, "cpu", spectral_norm=True, layer_norm=True)
    net_c1 = PlasticMLP(OBS_N, TINY, "cpu", spectral_norm=True, layer_norm=True)
    net_c2 = PlasticMLP(OBS_N, TINY, "cpu", spectral_norm=True, layer_norm=True)
    actor = MaskedSACActor(net_a, ACT_N, "cpu")
    critic1 = ObsExtractCritic(net_c1, ACT_N, "cpu")
    critic2 = ObsExtractCritic(net_c2, ACT_N, "cpu")
    return DiscreteSACPolicy(
        actor, torch.optim.Adam(actor.parameters()),
        critic1, torch.optim.Adam(critic1.parameters()),
        critic2, torch.optim.Adam(critic2.parameters()),
        tau=0.005, gamma=0.99, alpha=0.2, estimation_step=3,
    )


def _build_dqn():
    from tianshou.policy import DQNPolicy
    from dqn import QNet

    net = QNet(OBS_N, ACT_N, TINY, "cpu")
    policy = DQNPolicy(
        net, torch.optim.Adam(net.parameters()),
        discount_factor=0.99, estimation_step=3,
        target_update_freq=100, is_double=True,
    )
    policy.set_eps(0.0)
    return policy


def _build_qrdqn():
    from tianshou.policy import QRDQNPolicy
    from qrdqn import QRNet

    net = QRNet(OBS_N, ACT_N, TINY, num_quantiles=20, device="cpu")
    policy = QRDQNPolicy(
        net, torch.optim.Adam(net.parameters()),
        discount_factor=0.99, num_quantiles=20,
        estimation_step=3, target_update_freq=100,
    )
    policy.set_eps(0.0)
    return policy


def _build_iqn():
    from tianshou.policy import IQNPolicy
    from iqn import ObsExtractNet
    from tianshou.utils.net.discrete import ImplicitQuantileNetwork

    preprocess = ObsExtractNet(OBS_N, TINY, "cpu")
    net = ImplicitQuantileNetwork(
        preprocess, ACT_N, num_cosines=16,
        preprocess_net_output_dim=preprocess.output_dim, device="cpu",
    )
    policy = IQNPolicy(
        net, torch.optim.Adam(net.parameters()),
        discount_factor=0.99, sample_size=8,
        online_sample_size=4, target_sample_size=4,
        estimation_step=3, target_update_freq=100,
    )
    policy.set_eps(0.0)
    return policy


ALGO_BUILDERS = {
    "ppo": _build_ppo,
    "rainbow": _build_rainbow,
    "sac": _build_sac,
    "sac_v2": _build_sac_v2,
    "dqn": _build_dqn,
    "qrdqn": _build_qrdqn,
    "iqn": _build_iqn,
}


# ── Tests ───────────────────────────────────────────────────────────


class TestGetAction:
    """get_action must work for every algo without crashing."""

    @pytest.mark.parametrize("algo", list(ALGO_BUILDERS.keys()))
    def test_get_action_returns_int(self, algo):
        from eval import get_action
        policy = ALGO_BUILDERS[algo]()
        policy.eval()
        obs = _get_obs()
        action = get_action(policy, obs, algo)
        assert isinstance(action, int)
        assert 0 <= action < ACT_N

    @pytest.mark.parametrize("algo", list(ALGO_BUILDERS.keys()))
    def test_action_respects_mask(self, algo):
        from eval import get_action
        policy = ALGO_BUILDERS[algo]()
        policy.eval()
        obs = _get_obs()
        action = get_action(policy, obs, algo)
        assert obs["mask"][action], f"{algo} chose masked-out action {action}"


class TestBuildPolicy:
    """Builder functions must construct valid policies."""

    @pytest.mark.parametrize("algo", list(ALGO_BUILDERS.keys()))
    def test_builds_without_error(self, algo):
        policy = ALGO_BUILDERS[algo]()
        assert policy is not None


class TestSaveLoadRoundtrip:
    """save → load → get_action must work for algos with custom wrappers."""

    @pytest.mark.parametrize("algo", ["sac", "sac_v2", "ppo"])
    def test_roundtrip(self, algo, tmp_path):
        from eval import get_action
        policy = ALGO_BUILDERS[algo]()
        policy.eval()

        path = tmp_path / "test.pth"
        torch.save(policy.state_dict(), path)

        policy2 = ALGO_BUILDERS[algo]()
        policy2.load_state_dict(torch.load(path, map_location="cpu"))
        policy2.eval()

        obs = _get_obs()
        a1 = get_action(policy, obs, algo)
        a2 = get_action(policy2, obs, algo)
        assert a1 == a2, "Loaded policy should produce same action as original"
