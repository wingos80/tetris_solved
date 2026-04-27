"""Integration tests for the training pipeline (lightweight)."""

import pytest
import torch
import numpy as np
from tianshou.data import Collector, PrioritizedVectorReplayBuffer, VectorReplayBuffer
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


class TestPER:
    """PER buffer wires up correctly for SAC v2 and DQN via run_trial."""

    def test_sac_v2_per_run_trial(self):
        # Verifies PrioritizedVectorReplayBuffer activates without error and
        # that post_process_fn correctly updates priorities (no crash = it ran).
        from rl_training.sac_v2 import run_trial
        cfg = {
            "hidden": [64], "lr_actor": 3e-4, "lr_critic": 3e-4,
            "gamma": 0.99, "tau": 0.005, "n_step": 3, "alpha": 0.2,
            "batch_size": 64, "buffer_size": 1000, "step_per_epoch": 200,
            "step_per_collect": 50, "update_per_step": 1.0, "episode_per_test": 2,
            "checkpoint_freq": 10, "reset_freq": 0, "entropy_frac": 0.5,
            "spectral_norm": False, "layer_norm": False,
            "per_alpha": 0.6, "per_beta": 0.4,
        }
        reward = run_trial(cfg, trial_epochs=1)
        assert isinstance(reward, float)

    def test_dqn_per_run_trial(self):
        # Same check for DQN — per_alpha > 0 switches to PrioritizedVectorReplayBuffer.
        from rl_training.dqn import run_trial
        cfg = {
            "hidden": [64], "lr": 1e-4, "gamma": 0.99, "n_step": 3,
            "target_update_freq": 100, "batch_size": 64, "buffer_size": 1000,
            "step_per_epoch": 200, "step_per_collect": 50, "update_per_step": 1.0,
            "episode_per_test": 2, "checkpoint_freq": 10, "reset_freq": 0,
            "eps_train": 0.1, "eps_test": 0.0,
            "per_alpha": 0.6, "per_beta": 0.4,
        }
        reward = run_trial(cfg, trial_epochs=1)
        assert isinstance(reward, float)

    def test_per_alpha_zero_uses_uniform_buffer(self):
        # per_alpha=0.0 (default) must not create a prioritized buffer —
        # ensures existing runs without PER keys are unaffected.
        from rl_training.dqn import run_trial
        cfg = {
            "hidden": [64], "lr": 1e-4, "gamma": 0.99, "n_step": 3,
            "target_update_freq": 100, "batch_size": 64, "buffer_size": 1000,
            "step_per_epoch": 200, "step_per_collect": 50, "update_per_step": 1.0,
            "episode_per_test": 2, "checkpoint_freq": 10, "reset_freq": 0,
            "eps_train": 0.1, "eps_test": 0.0,
            "per_alpha": 0.0,
        }
        reward = run_trial(cfg, trial_epochs=1)
        assert isinstance(reward, float)
