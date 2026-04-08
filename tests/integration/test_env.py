"""Integration tests for the Gymnasium environment."""

import numpy as np
import pytest
from tetris import Tetris
from tetris.env import TetrisEnv


@pytest.fixture
def env():
    e = TetrisEnv()
    e.reset()
    return e


class TestReset:
    def test_obs_is_dict(self, env):
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert "obs" in obs and "mask" in obs

    def test_obs_shape(self, env):
        obs, _ = env.reset()
        assert obs["obs"].shape == (78,)
        assert obs["mask"].shape == (40,)

    def test_obs_range(self, env):
        obs, _ = env.reset()
        assert obs["obs"].min() >= -1.0
        assert obs["obs"].max() <= 1.0

    def test_mask_has_valid_actions(self, env):
        obs, _ = env.reset()
        assert obs["mask"].any()


class TestStep:
    def test_valid_action(self, env):
        obs, _ = env.reset()
        action = obs["mask"].nonzero()[0][0]
        obs, reward, term, trunc, info = env.step(action)
        assert info["valid"] is True
        assert obs["obs"].shape == (78,)

    def test_invalid_action(self, env):
        obs, _ = env.reset()
        invalid = (~obs["mask"].astype(bool)).nonzero()[0]
        if len(invalid) == 0:
            pytest.skip("All actions valid")
        obs, reward, term, trunc, info = env.step(invalid[0])
        assert info["valid"] is False
        assert reward == -1.0

    def test_no_truncation(self, env):
        obs, _ = env.reset()
        action = obs["mask"].nonzero()[0][0]
        _, _, _, trunc, _ = env.step(action)
        assert trunc is False


class TestActionMasks:
    def test_shape(self, env):
        obs, _ = env.reset()
        assert obs["mask"].shape == (40,)

    def test_at_least_one_valid(self, env):
        obs, _ = env.reset()
        assert obs["mask"].any()

    def test_masked_actions_are_consistent(self, env):
        """Every action marked valid should produce a valid placement."""
        obs, _ = env.reset()
        mask = obs["mask"].astype(bool)
        for a in np.where(mask)[0][:5]:  # test first 5 valid actions
            env.reset()
            _, _, _, _, info = env.step(int(a))
            assert info["valid"], f"Action {a} was masked valid but step said invalid"


class TestEpisode:
    def test_episode_terminates(self):
        env = TetrisEnv()
        obs, _ = env.reset()
        for _ in range(500):
            mask = obs["mask"].astype(bool)
            if mask.any():
                action = np.random.choice(np.where(mask)[0])
            else:
                action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            if term:
                return
        pytest.fail("Episode didn't terminate in 500 steps")

    def test_score_nonnegative(self):
        env = TetrisEnv()
        obs, _ = env.reset()
        for _ in range(100):
            mask = obs["mask"].astype(bool)
            action = np.random.choice(np.where(mask)[0]) if mask.any() else 0
            obs, _, term, _, info = env.step(action)
            if term:
                break
        assert info.get("score", 0) >= 0

    def test_multiple_resets(self):
        env = TetrisEnv()
        for _ in range(5):
            obs, _ = env.reset()
            assert obs["obs"].shape == (78,)
            mask = obs["mask"].astype(bool)
            action = np.random.choice(np.where(mask)[0])
            env.step(action)
