"""Unit tests for QR-DQN network architecture."""

import pytest
import torch
from rl_training.qrdqn import QRNet


class TestQRNet:
    def test_output_shape(self):
        net = QRNet(obs_n=78, act_n=40, hidden=[64, 64],
                    num_quantiles=200, device="cpu")
        obs = torch.randn(4, 78)
        out, state = net(obs)
        assert out.shape == (4, 40, 200)

    def test_output_shape_different_quantiles(self):
        net = QRNet(obs_n=78, act_n=40, hidden=[64, 64],
                    num_quantiles=50, device="cpu")
        obs = torch.randn(2, 78)
        out, _ = net(obs)
        assert out.shape == (2, 40, 50)

    def test_single_obs(self):
        net = QRNet(obs_n=78, act_n=40, hidden=[32],
                    num_quantiles=100, device="cpu")
        obs = torch.randn(1, 78)
        out, _ = net(obs)
        assert out.shape == (1, 40, 100)

    def test_dict_obs_extraction(self):
        """QRNet should handle tianshou dict obs (obs.obs attribute)."""
        net = QRNet(obs_n=78, act_n=40, hidden=[32],
                    num_quantiles=50, device="cpu")

        class DictObs:
            pass
        o = DictObs()
        o.obs = torch.randn(2, 78)
        o.mask = torch.ones(2, 40, dtype=torch.int8)

        out, _ = net(o)
        assert out.shape == (2, 40, 50)

    def test_q_values_from_mean(self):
        """Mean across quantiles should give Q-values for action selection."""
        net = QRNet(obs_n=78, act_n=40, hidden=[32],
                    num_quantiles=200, device="cpu")
        obs = torch.randn(3, 78)
        out, _ = net(obs)
        q_values = out.mean(dim=2)
        assert q_values.shape == (3, 40)

    def test_reset_output_head(self):
        """Output head reset should change parameters."""
        net = QRNet(obs_n=78, act_n=40, hidden=[32],
                    num_quantiles=50, device="cpu")
        obs = torch.randn(1, 78)
        out_before, _ = net(obs)
        net.net[-1].reset_parameters()
        out_after, _ = net(obs)
        assert not torch.allclose(out_before, out_after)
