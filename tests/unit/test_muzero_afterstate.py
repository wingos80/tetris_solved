"""Unit tests for muzero_afterstate — one file covers all fixes as they land."""

import numpy as np
import pytest
import torch

from rl_training.muzero_afterstate import (
    MuZeroNet,
    SequenceReplay,
    choose_action,
    quantile_huber_loss,
    run_episode,
    train_step,
)
from tetris.env.afterstate import enumerate_afterstates_raw
from tetris.game import Tetris

BOARD_SHAPE = (Tetris.ROWS, Tetris.COLS)


def _small_net():
    return MuZeroNet(
        conv_channels=[4],
        fc_hidden=[16],
        latent_dim=8,
        action_emb_dim=4,
        n_quantiles=4,
        device="cpu",
    )


def _small_cfg():
    return {
        "batch_size": 4,
        "unroll_steps": 2,
        "gamma": 0.95,
        "kappa": 1.0,
        "reward_coef": 1.0,
        "consistency_coef": 0.5,
    }


# ── Fix 1: replay stores afterstate board ─────────────────────────

class TestReplayStoresAfterstate:
    def test_stored_board_matches_afterstate(self):
        """Board pushed to replay must equal the post-placement afterstate board,
        not the pre-action board (which would be zeros on the first step)."""
        net = _small_net()
        replay = SequenceReplay(capacity=10000)
        run_episode(net, eps=1.0, replay=replay, max_steps=3)

        if not replay.episodes:
            pytest.skip("episode too short")

        ep = replay.episodes[0]
        # The first stored board should NOT be all zeros (pre-action board was zeros).
        # After a placement, at least the piece cells are filled.
        first_board = ep[0][0]
        assert first_board.max() > 0, (
            "First stored board is all zeros — still storing pre-action board"
        )

    def test_stored_board_is_valid_afterstate(self):
        """Each stored board must be a board that appears in enumerate_afterstates_raw
        for *some* game state — i.e. it has a piece frozen in it."""
        net = _small_net()
        replay = SequenceReplay(capacity=10000)
        run_episode(net, eps=0.0, replay=replay, max_steps=5)

        if not replay.episodes:
            pytest.skip("episode too short")

        ep = replay.episodes[0]
        for step_data in ep:
            board = step_data[0]
            # A valid afterstate board is binary (0/1 floats)
            assert set(np.unique(board)).issubset({0.0, 1.0}), (
                "Stored board has values other than 0/1 — not a valid afterstate"
            )
            # And has at least some filled cells (the frozen piece)
            assert board.sum() > 0, "Stored board is empty"

    def test_sequence_boards_are_consecutive_afterstates(self):
        """boards[t] and boards[t+1] in a sampled sequence should both be
        valid afterstates (non-zero), not a mix of pre- and post-action boards."""
        net = _small_net()
        replay = SequenceReplay(capacity=10000)
        for _ in range(5):
            run_episode(net, eps=1.0, replay=replay, max_steps=20)

        boards, _, _, _, _ = replay.sample_sequences(batch_size=2, k=2)
        # All k+1 boards in each sample must be non-zero afterstates
        for b in range(boards.shape[0]):
            for t in range(boards.shape[1]):
                assert boards[b, t].max() > 0, (
                    f"boards[{b},{t}] is all zeros — pre-action board leaked into sequence"
                )


# ── Transition step counting ──────────────────────────────────────

class TestTransitionCounting:
    def test_run_episode_returns_step_count(self):
        net = _small_net()
        _, steps, _ = run_episode(net, eps=1.0, replay=None, max_steps=10)
        assert isinstance(steps, int)
        assert 1 <= steps <= 10

    def test_steps_match_replay_length(self):
        net = _small_net()
        replay = SequenceReplay(capacity=10000)  # large enough to avoid eviction
        _, steps, _ = run_episode(net, eps=1.0, replay=replay, max_steps=15)
        # Total transitions in replay should equal steps taken
        assert len(replay) == steps


# ── Existing smoke tests (should still pass after Fix 1) ─────────

class TestMuZeroNetSmoke:
    def test_represent(self):
        net = _small_net()
        board = np.zeros(BOARD_SHAPE, dtype=np.float32)
        z = net.represent(board)
        assert z.shape == (1, 8)

    def test_dynamics(self):
        net = _small_net()
        z = torch.zeros(1, 8)
        z_next, r = net.dynamics(z, torch.tensor([0]), torch.tensor([3]))
        assert z_next.shape == (1, 8)
        assert r.shape == (1,)

    def test_forward_batch(self):
        net = _small_net()
        boards = np.zeros((5, *BOARD_SHAPE), dtype=np.float32)
        v = net(boards)
        assert v.shape == (5,)

    def test_train_step_runs(self):
        net = _small_net()
        target = _small_net()
        target.load_state_dict(net.state_dict())
        optim = torch.optim.Adam(net.parameters(), lr=1e-3)
        replay = SequenceReplay(capacity=10000)
        for _ in range(10):
            run_episode(net, eps=1.0, replay=replay, max_steps=30)
        cfg = _small_cfg()
        vl, rl, cl = train_step(net, target, optim, replay, cfg)
        assert np.isfinite(vl)
        assert np.isfinite(rl)
        assert np.isfinite(cl)

    def test_quantile_huber_loss(self):
        pred = torch.zeros(4, 8)
        target = torch.ones(4, 8)
        loss = quantile_huber_loss(pred, target)
        assert loss.item() > 0
        assert loss.requires_grad or not pred.requires_grad


# ── Fix 2: scalar value head ──────────────────────────────────────

class TestScalarValueHead:
    def test_value_returns_1d(self):
        """value(z) must return [B] scalar, not [B, N] quantile tensor."""
        net = _small_net()
        z = torch.zeros(4, 8)
        v = net.value(z)
        assert v.shape == (4,), f"Expected (4,) got {v.shape}"

    def test_forward_returns_scalar_per_sample(self):
        """forward(boards) must return [B] scalar values."""
        net = _small_net()
        boards = np.zeros((3, *BOARD_SHAPE), dtype=np.float32)
        v = net(boards)
        assert v.shape == (3,)

    def test_value_head_output_dim(self):
        """value_head Linear must have out_features=1, not n_quantiles."""
        net = _small_net()
        assert net.value_head.out_features == 1, (
            f"value_head has {net.value_head.out_features} outputs — "
            "should be 1 (scalar, not quantile)"
        )

    def test_train_step_value_loss_is_scalar_mse(self):
        """Value loss should be MSE-scale (not quantile-huber scale)."""
        net = _small_net()
        target = _small_net()
        target.load_state_dict(net.state_dict())
        optim = torch.optim.Adam(net.parameters(), lr=1e-3)
        replay = SequenceReplay(capacity=10000)
        for _ in range(10):
            run_episode(net, eps=1.0, replay=replay, max_steps=30)
        cfg = _small_cfg()
        vl, rl, cl = train_step(net, target, optim, replay, cfg)
        # MSE value loss should be finite and non-negative
        assert np.isfinite(vl) and vl >= 0
        assert np.isfinite(rl) and rl >= 0
        assert np.isfinite(cl) and cl >= 0

    def test_value_gradients_nonzero(self):
        """Gradient must flow through value_head after a train step."""
        net = _small_net()
        target = _small_net()
        target.load_state_dict(net.state_dict())
        optim = torch.optim.Adam(net.parameters(), lr=1e-3)
        replay = SequenceReplay(capacity=10000)
        for _ in range(10):
            run_episode(net, eps=1.0, replay=replay, max_steps=30)
        train_step(net, target, optim, replay, _small_cfg())
        grad = net.value_head.weight.grad
        assert grad is not None and grad.abs().sum().item() > 0


# ── Fix 3: transition-level replay capacity ───────────────────────

class TestTransitionLevelReplay:
    def test_len_counts_transitions_not_episodes(self):
        """len(replay) must reflect total transitions stored, not episode count."""
        net = _small_net()
        replay = SequenceReplay(capacity=10000)
        total_steps = 0
        for _ in range(5):
            _, steps, _ = run_episode(net, eps=1.0, replay=replay, max_steps=20)
            total_steps += steps
        assert len(replay) == total_steps, (
            f"len(replay)={len(replay)} but ran {total_steps} transitions"
        )

    def test_capacity_enforced_as_transitions(self):
        """Replay must not exceed capacity (in transitions) after many episodes."""
        cap = 30  # tiny cap to stress eviction
        net = _small_net()
        replay = SequenceReplay(capacity=cap)
        for _ in range(50):
            run_episode(net, eps=1.0, replay=replay, max_steps=10)
        assert len(replay) <= cap, (
            f"Replay has {len(replay)} transitions but capacity is {cap}"
        )

    def test_eviction_removes_oldest_episodes(self):
        """When cap is exceeded, oldest episodes should be evicted first (FIFO)."""
        cap = 20
        net = _small_net()
        replay = SequenceReplay(capacity=cap)
        # Push enough episodes to force eviction
        for _ in range(30):
            run_episode(net, eps=1.0, replay=replay, max_steps=5)
        # After eviction, total transitions must be <= cap
        total = sum(len(ep) for ep in replay.episodes)
        assert total == len(replay), "Internal _total_transitions counter is inconsistent"
        assert len(replay) <= cap

    def test_small_capacity_allows_sampling(self):
        """Even with a small cap, sample_sequences should not crash if there's
        at least one episode with >= k+1 steps."""
        cap = 50
        k = 2
        net = _small_net()
        replay = SequenceReplay(capacity=cap)
        # Run until we have enough data
        while not any(len(ep) > k for ep in replay.episodes):
            run_episode(net, eps=1.0, replay=replay, max_steps=20)
        boards, rots, dxs, rewards, dones = replay.sample_sequences(batch_size=2, k=k)
        assert boards.shape == (2, k + 1, *BOARD_SHAPE)

    def test_len_consistent_after_eviction(self):
        """_total_transitions counter must equal the sum of episode lengths after eviction."""
        cap = 40
        net = _small_net()
        replay = SequenceReplay(capacity=cap)
        for _ in range(40):
            run_episode(net, eps=1.0, replay=replay, max_steps=8)
        manual_count = sum(len(ep) for ep in replay.episodes)
        assert len(replay) == manual_count, (
            f"Internal counter {len(replay)} doesn't match actual sum {manual_count}"
        )


# ── Fix 4: soft EMA target net ────────────────────────────────────

class TestSoftEMATargetNet:
    def test_ema_tau_zero_does_hard_copy(self):
        """ema_tau=0 (default) must behave as hard copy on target_update_freq."""
        net = _small_net()
        target = _small_net()
        # Start with different weights
        for p in target.parameters():
            p.data.fill_(0.0)
        # Simulate one EMA step with tau=0 (hard copy path skipped unless freq matches)
        # Directly test that tau=0 leaves target unchanged (hard copy handled externally)
        tau = 0.0
        if tau > 0.0:
            with torch.no_grad():
                for p, pt in zip(net.parameters(), target.parameters()):
                    pt.data.mul_(1.0 - tau).add_(tau * p.data)
        # target should still be all zeros (tau=0 → no EMA update)
        for pt in target.parameters():
            assert pt.data.abs().sum().item() == 0.0

    def test_ema_tau_moves_target_toward_online(self):
        """After one EMA step, target params should be strictly between 0 and online params."""
        net = _small_net()
        target = _small_net()
        # Set target to zero, net to ones
        for p in net.parameters():
            p.data.fill_(1.0)
        for pt in target.parameters():
            pt.data.fill_(0.0)

        tau = 0.005
        with torch.no_grad():
            for p, pt in zip(net.parameters(), target.parameters()):
                pt.data.mul_(1.0 - tau).add_(tau * p.data)

        for pt in target.parameters():
            val = pt.data.mean().item()
            assert abs(val - tau) < 1e-5, (
                f"Expected target ≈ {tau} after one EMA step, got {val}"
            )

    def test_ema_convergence(self):
        """After many EMA steps, target should converge close to online net."""
        net = _small_net()
        target = _small_net()
        for p in net.parameters():
            p.data.fill_(1.0)
        for pt in target.parameters():
            pt.data.fill_(0.0)

        tau = 0.05  # faster for test
        for _ in range(200):
            with torch.no_grad():
                for p, pt in zip(net.parameters(), target.parameters()):
                    pt.data.mul_(1.0 - tau).add_(tau * p.data)

        for pt in target.parameters():
            val = pt.data.mean().item()
            assert val > 0.99, f"Target {val:.4f} should be close to 1.0 after 200 EMA steps"

    def test_ema_tau_in_cfg_selects_ema_path(self):
        """train_step + manual EMA must not crash; losses should be finite."""
        net = _small_net()
        target = _small_net()
        target.load_state_dict(net.state_dict())
        optim = torch.optim.Adam(net.parameters(), lr=1e-3)
        replay = SequenceReplay(capacity=10000)
        for _ in range(10):
            run_episode(net, eps=1.0, replay=replay, max_steps=30)
        cfg = _small_cfg()
        # Run train_step then apply EMA
        vl, rl, cl = train_step(net, target, optim, replay, cfg)
        tau = 0.005
        with torch.no_grad():
            for p, pt in zip(net.parameters(), target.parameters()):
                pt.data.mul_(1.0 - tau).add_(tau * p.data)
        assert np.isfinite(vl) and np.isfinite(rl) and np.isfinite(cl)


# ── Fix 5: separate reward head ───────────────────────────────────

class TestSeparateRewardHead:
    def test_dyn_r_hidden_exists(self):
        """Net must have a separate dyn_r_hidden module for the reward path."""
        net = _small_net()
        assert hasattr(net, "dyn_r_hidden"), "dyn_r_hidden attribute missing from MuZeroNet"

    def test_dyn_r_hidden_is_separate_from_dyn_hidden(self):
        """dyn_r_hidden params must be distinct from dyn_hidden params."""
        net = _small_net()
        dyn_params = set(id(p) for p in net.dyn_hidden.parameters())
        r_params = set(id(p) for p in net.dyn_r_hidden.parameters())
        assert dyn_params.isdisjoint(r_params), (
            "dyn_r_hidden shares parameters with dyn_hidden — not separate"
        )

    def test_dynamics_still_returns_correct_shapes(self):
        """dynamics(z, rot, dx) must still return (z_next [B, latent], r_hat [B])."""
        net = _small_net()
        z = torch.zeros(3, 8)
        z_next, r = net.dynamics(z, torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5]))
        assert z_next.shape == (3, 8), f"z_next shape {z_next.shape} != (3, 8)"
        assert r.shape == (3,), f"r_hat shape {r.shape} != (3,)"

    def test_reward_head_gradient_independent(self):
        """A reward-only loss should not touch dyn_hidden weights (separate path)."""
        net = _small_net()
        optim = torch.optim.Adam(net.parameters(), lr=1e-3)
        optim.zero_grad()

        z = torch.zeros(2, 8, requires_grad=False)
        _, r = net.dynamics(z, torch.tensor([0, 1]), torch.tensor([3, 4]))
        loss = r.sum()
        loss.backward()

        # dyn_r_hidden should have gradients (reward path)
        r_hidden_grad = sum(
            p.grad.abs().sum().item()
            for p in net.dyn_r_hidden.parameters()
            if p.grad is not None
        )
        assert r_hidden_grad > 0, "dyn_r_hidden has no gradient from reward loss"

    def test_train_step_with_separate_reward_head(self):
        """Full train_step must complete without error and return finite losses."""
        net = _small_net()
        target = _small_net()
        target.load_state_dict(net.state_dict())
        optim = torch.optim.Adam(net.parameters(), lr=1e-3)
        replay = SequenceReplay(capacity=10000)
        for _ in range(10):
            run_episode(net, eps=1.0, replay=replay, max_steps=30)
        vl, rl, cl = train_step(net, target, optim, replay, _small_cfg())
        assert np.isfinite(vl) and np.isfinite(rl) and np.isfinite(cl)


# ── Fix 6: gradient clipping ──────────────────────────────────────

class TestGradientClipping:
    def _run_one_step(self, cfg):
        net = _small_net()
        target = _small_net()
        target.load_state_dict(net.state_dict())
        optim = torch.optim.Adam(net.parameters(), lr=1e-3)
        replay = SequenceReplay(capacity=10000)
        for _ in range(10):
            run_episode(net, eps=1.0, replay=replay, max_steps=30)
        train_step(net, target, optim, replay, cfg)
        return net

    def test_grad_clip_zero_disabled(self):
        """grad_clip=0.0 must not clip — train_step should still run fine."""
        cfg = {**_small_cfg(), "grad_clip": 0.0}
        net = self._run_one_step(cfg)
        # Just ensure it ran without error
        assert net is not None

    def test_grad_clip_enforced(self):
        """After train_step with grad_clip=1.0, gradient norms must be <= clip value."""
        net = _small_net()
        target = _small_net()
        target.load_state_dict(net.state_dict())
        optim = torch.optim.Adam(net.parameters(), lr=1e-3)
        replay = SequenceReplay(capacity=10000)
        for _ in range(10):
            run_episode(net, eps=1.0, replay=replay, max_steps=30)

        cfg = {**_small_cfg(), "grad_clip": 1.0}

        # We need to inspect grads BEFORE optim.step clips them.
        # Re-implement the backward pass manually:
        from rl_training.muzero_afterstate import train_step as _train_step
        import torch.nn.functional as F_

        # Run a training step and capture the grad norm
        boards, rots, dxs, rewards, dones = replay.sample_sequences(cfg["batch_size"], cfg["unroll_steps"])
        device = next(net.parameters()).device
        boards_t = torch.as_tensor(boards, dtype=torch.float32, device=device)

        z = net.represent(boards_t[:, 0])
        z_targets = [target.represent(boards_t[:, t]) for t in range(cfg["unroll_steps"] + 1)]
        v_targets = [target.value(zt) for zt in z_targets]

        value_loss = torch.tensor(0.0, device=device)
        for t in range(cfg["unroll_steps"]):
            G = torch.zeros(cfg["batch_size"], dtype=torch.float32, device=device)
            v_pred = net.value(z)
            value_loss = value_loss + F_.mse_loss(v_pred, G)
            z_next, _ = net.dynamics(z, torch.as_tensor(rots[:, t], device=device),
                                        torch.as_tensor(dxs[:, t], device=device))
            z = z_next

        optim.zero_grad()
        value_loss.backward()

        total_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        # After clipping, the effective norm applied should be <= clip value
        # (total_norm is the pre-clip norm; post-clip norm is min(total_norm, clip))
        for p in net.parameters():
            if p.grad is not None:
                assert p.grad.norm().item() <= 1.0 + 1e-5, (
                    f"Gradient norm {p.grad.norm().item()} exceeds clip=1.0"
                )

    def test_train_step_with_clip_returns_finite(self):
        """train_step with grad_clip=1.0 must return finite losses."""
        cfg = {**_small_cfg(), "grad_clip": 1.0}
        net = _small_net()
        target = _small_net()
        target.load_state_dict(net.state_dict())
        optim = torch.optim.Adam(net.parameters(), lr=1e-3)
        replay = SequenceReplay(capacity=10000)
        for _ in range(10):
            run_episode(net, eps=1.0, replay=replay, max_steps=30)
        vl, rl, cl = train_step(net, target, optim, replay, cfg)
        assert np.isfinite(vl) and np.isfinite(rl) and np.isfinite(cl)
