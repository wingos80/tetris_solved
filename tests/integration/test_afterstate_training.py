"""Integration tests for afterstate DQN training (lightweight)."""

import torch
import numpy as np

from rl_training.afterstate_dqn import (
    Replay,
    VNet,
    choose_action,
    evaluate,
    run_episode,
    run_trial,
    train_step,
)
from tetris.game import Tetris


class TestVNet:
    def test_forward_single_state(self):
        net = VNet([16, 16], "cpu")
        x = np.zeros(4, dtype=np.float32)
        v = net(x)
        assert v.shape == ()  # scalar after squeeze

    def test_forward_batch(self):
        net = VNet([16, 16], "cpu")
        x = np.zeros((8, 4), dtype=np.float32)
        v = net(x)
        assert v.shape == (8,)


class TestReplay:
    def test_push_and_sample(self):
        buf = Replay(100)
        for i in range(20):
            s = np.array([i, 0, 0, 0], dtype=np.float32)
            s_next = np.array([i + 1, 0, 0, 0], dtype=np.float32)
            buf.push(s, s_next, float(i), False)
        s, s_next, r, done = buf.sample(8)
        assert s.shape == (8, 4)
        assert s_next.shape == (8, 4)
        assert r.shape == (8,)
        assert done.shape == (8,)

    def test_capacity_evicts_oldest(self):
        buf = Replay(5)
        for i in range(10):
            buf.push(np.zeros(4, dtype=np.float32),
                     np.zeros(4, dtype=np.float32),
                     float(i), False)
        assert len(buf) == 5


class TestChooseAction:
    def test_returns_valid_action_or_none(self):
        net = VNet([16, 16], "cpu")
        game = Tetris()
        game.next_block()
        action, feats = choose_action(game, net, eps=0.0)
        assert action is not None
        assert feats is not None
        assert feats.shape == (4,)

    def test_eps_one_is_random(self):
        # With eps=1, should still return a valid action from enumeration
        net = VNet([16, 16], "cpu")
        game = Tetris()
        game.next_block()
        action, feats = choose_action(game, net, eps=1.0)
        assert action is not None

    def test_returns_none_when_gameover(self):
        net = VNet([16, 16], "cpu")
        game = Tetris()
        game.next_block()
        game.state = "gameover"
        action, feats = choose_action(game, net, eps=0.0)
        assert action is None


class TestRunEpisode:
    def test_episode_terminates(self):
        net = VNet([16, 16], "cpu")
        replay = Replay(10000)
        reward, steps, lines = run_episode(net, eps=1.0, replay=replay,
                                           gamma=0.95, max_steps=500)
        assert steps > 0
        assert len(replay) == steps
        assert lines >= 0

    def test_replay_captures_transitions(self):
        net = VNet([16, 16], "cpu")
        replay = Replay(10000)
        run_episode(net, eps=1.0, replay=replay, gamma=0.95, max_steps=20)
        s, s_next, r, done = replay.sample(min(8, len(replay)))
        # Last transition should have done=True (episode ended)
        # Note: only true if game actually ended within max_steps
        assert s.shape[1] == 4


class TestTrainStep:
    def test_loss_decreases_on_constant_target(self):
        """Smoke test: net should overfit a single tiny batch."""
        net = VNet([32, 32], "cpu")
        target_net = VNet([32, 32], "cpu")
        target_net.load_state_dict(net.state_dict())
        optim = torch.optim.Adam(net.parameters(), lr=1e-2)
        replay = Replay(100)
        # Push fixed transitions: all zeros -> target ~= reward (s_next zeros, gamma * V(0))
        for _ in range(64):
            replay.push(np.zeros(4, dtype=np.float32),
                        np.zeros(4, dtype=np.float32),
                        5.0, True)  # done=True so target = r = 5
        losses = [train_step(net, target_net, optim, replay,
                             batch_size=32, gamma=0.95) for _ in range(50)]
        assert losses[-1] < losses[0]


class TestEvaluate:
    def test_evaluate_returns_floats(self):
        net = VNet([16, 16], "cpu")
        mean_r, mean_lines, mean_steps = evaluate(net, n_episodes=2)
        assert isinstance(mean_r, float)
        assert isinstance(mean_lines, float)
        assert isinstance(mean_steps, float)


class TestRunTrial:
    def test_short_trial_completes(self):
        cfg = {
            "hidden": [16, 16],
            "lr": 1e-3,
            "gamma": 0.95,
            "batch_size": 16,
            "buffer_size": 1000,
            "updates_per_episode": 1,
            "target_update_freq": 5,
            "eps_start": 1.0,
            "eps_end": 0.0,
            "eps_decay_episodes": 5,
            "eval_freq": 5,
            "eval_episodes": 2,
        }
        best = run_trial(cfg, trial_episodes=10)
        assert isinstance(best, float)
