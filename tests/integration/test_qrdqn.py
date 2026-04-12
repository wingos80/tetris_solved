"""Integration tests for QR-DQN training pipeline."""

import pytest
import torch
import numpy as np
from tianshou.data import Batch, Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import QRDQNPolicy

from tetris.env import TetrisEnv
from rl_training.qrdqn import QRNet, load_config, run_trial


class TestQRDQNPolicy:
    def test_policy_construction(self):
        net = QRNet(78, 40, [64, 64], num_quantiles=50, device="cpu")
        optim = torch.optim.Adam(net.parameters())
        policy = QRDQNPolicy(
            net, optim,
            discount_factor=0.99,
            num_quantiles=50,
            estimation_step=3,
            target_update_freq=100,
        )
        assert policy is not None

    def test_action_masking(self):
        """QR-DQN inherits DQN masking — invalid actions should never be chosen."""
        net = QRNet(78, 40, [64, 64], num_quantiles=50, device="cpu")
        optim = torch.optim.Adam(net.parameters())
        policy = QRDQNPolicy(
            net, optim,
            discount_factor=0.99,
            num_quantiles=50,
            estimation_step=1,
            target_update_freq=100,
        )
        policy.set_eps(0.0)

        env = TetrisEnv()
        obs, _ = env.reset()
        obs_t = torch.as_tensor(obs["obs"], dtype=torch.float32).unsqueeze(0)
        mask_int = torch.as_tensor(obs["mask"], dtype=torch.int8).unsqueeze(0)

        batch = Batch(obs=Batch(obs=obs_t, mask=mask_int), info={})
        with torch.no_grad():
            result = policy(batch)
        action = result.act.item()
        assert obs["mask"][action] == 1, "Policy chose an invalid (masked) action"

    def test_collect_steps(self):
        net = QRNet(78, 40, [32], num_quantiles=20, device="cpu")
        optim = torch.optim.Adam(net.parameters())
        policy = QRDQNPolicy(
            net, optim,
            discount_factor=0.99,
            num_quantiles=20,
            estimation_step=1,
            target_update_freq=100,
        )
        policy.set_eps(1.0)

        envs = DummyVectorEnv([lambda: TetrisEnv() for _ in range(2)])
        collector = Collector(policy, envs, VectorReplayBuffer(500, 2),
                              exploration_noise=True)
        result = collector.collect(n_step=50)
        assert result["n/st"] >= 50
        envs.close()


class TestQRDQNConfig:
    def test_load_default_config(self):
        cfg = load_config()
        assert "hidden" in cfg
        assert "num_quantiles" in cfg
        assert "lr" in cfg
        assert cfg["num_quantiles"] == 51

    def test_load_override(self, tmp_path):
        override = tmp_path / "override.json"
        override.write_text('{"num_quantiles": 100, "lr": 0.001}')
        cfg = load_config(str(override))
        assert cfg["num_quantiles"] == 100
        assert cfg["lr"] == 0.001
        # defaults should remain for non-overridden keys
        assert "hidden" in cfg


class TestQRDQNTrial:
    @pytest.mark.slow
    def test_run_trial_returns_float(self):
        """Smoke test: run_trial completes and returns a numeric reward."""
        cfg = load_config()
        cfg["hidden"] = [32]
        cfg["num_quantiles"] = 20
        cfg["buffer_size"] = 1000
        cfg["step_per_epoch"] = 200
        cfg["step_per_collect"] = 50
        cfg["episode_per_test"] = 2
        cfg["batch_size"] = 32
        result = run_trial(cfg, trial_epochs=1)
        assert isinstance(result, float)
