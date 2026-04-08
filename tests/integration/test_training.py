"""Integration tests for the training pipeline (lightweight)."""

import pytest
import torch
import numpy as np
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from torch.distributions import Categorical

from tetris.env import TetrisEnv
from rl_training.ppo import MaskedActor, ObsExtractCritic


def _make_policy():
    from tianshou.policy import PPOPolicy
    from tianshou.utils.net.common import Net
    from tianshou.utils.net.discrete import Actor, Critic

    obs_n, act_n = 78, 40
    net_a = Net(state_shape=obs_n, hidden_sizes=[32, 32], device="cpu")
    net_c = Net(state_shape=obs_n, hidden_sizes=[32, 32], device="cpu")
    raw_actor = Actor(net_a, act_n, device="cpu", softmax_output=False)
    raw_critic = Critic(net_c, device="cpu")
    actor = MaskedActor(raw_actor)
    critic = ObsExtractCritic(raw_critic)

    return PPOPolicy(
        actor, critic,
        torch.optim.Adam(list(actor.parameters()) + list(critic.parameters())),
        dist_fn=lambda logits: Categorical(logits=logits),
        action_space=TetrisEnv().action_space,
    )


class TestMaskedActor:
    def test_masked_logits(self):
        policy = _make_policy()
        env = TetrisEnv()
        obs, _ = env.reset()
        obs_t = torch.as_tensor(obs["obs"], dtype=torch.float32).unsqueeze(0)
        mask_t = torch.as_tensor(obs["mask"], dtype=torch.bool).unsqueeze(0)

        class Obs:
            pass
        o = Obs()
        o.obs = obs_t
        o.mask = mask_t

        with torch.no_grad():
            logits, _ = policy.actor(o)
        # Invalid actions should have very low logits
        invalid = ~mask_t.squeeze()
        if invalid.any():
            assert logits[0, invalid].max() < -1e7


class TestCollector:
    def test_collect_steps(self):
        policy = _make_policy()
        envs = DummyVectorEnv([lambda: TetrisEnv() for _ in range(2)])
        collector = Collector(policy, envs, VectorReplayBuffer(1000, 2))
        result = collector.collect(n_step=50)
        assert result["n/st"] >= 50

    def test_collect_episodes(self):
        policy = _make_policy()
        envs = DummyVectorEnv([lambda: TetrisEnv() for _ in range(2)])
        collector = Collector(policy, envs, VectorReplayBuffer(1000, 2))
        result = collector.collect(n_episode=2)
        assert result["n/ep"] >= 2
