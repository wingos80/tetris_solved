"""Gymnasium environment — placement action space over Tetris."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tetris.game import Tetris, NUM_BLOCK_TYPES
from tetris.env import rewards


class TetrisEnv(gym.Env):
    """Placement action space: Discrete(4 * COLS = 40).

    action = rotation * COLS + column, where column is block.x.
    Rotations beyond a piece's max wrap via modulo.
    Invalid placements receive a small penalty and no state change.

    Observation dict:
        obs (78-d float32): column heights (10) | holes per col (10) |
            height diffs (9) | current piece one-hot (7) | hold piece
            one-hot (7) | next-5 pieces one-hot (35)
        mask (40-d int8): 1 = valid placement, 0 = invalid
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.game = Tetris()
        n_act = 4 * Tetris.COLS
        self.action_space = spaces.Discrete(n_act)
        self.observation_space = spaces.Dict({
            "obs": spaces.Box(low=-1.0, high=1.0, shape=(78,), dtype=np.float32),
            "mask": spaces.MultiBinary(n_act),
        })
        self._prev_height = 0
        self._prev_holes = 0
        self._prev_bumpiness = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game = Tetris()
        self.game.next_block()
        self._prev_height = 0
        self._prev_holes = 0
        self._prev_bumpiness = 0
        return self._obs(), {}

    def step(self, action):
        block = self.game.block
        rot = (action // Tetris.COLS) % block.num_rotations
        col = action % Tetris.COLS

        # Try placement at top of field
        block.rotation, block.x, block.y = rot, col, 0

        if self.game._intersects():
            # Restore to spawn position so env stays usable
            block.rotation, block.x, block.y = 0, Tetris.COLS // 2 - 2, 0
            return self._obs(), -1.0, False, False, {"lines": 0, "valid": False}

        lines = self.game.hard_drop()
        height, holes, bumpiness = self._height_holes_bumpiness()
        game_over = self.game.state == "gameover"

        reward = rewards.compute(
            lines, game_over,
            height - self._prev_height, holes - self._prev_holes,
            bumpiness - self._prev_bumpiness,
        )
        self._prev_height = height
        self._prev_holes = holes
        self._prev_bumpiness = bumpiness

        return self._obs(), reward, game_over, False, {
            "lines": lines, "valid": True, "score": self.game.score,
        }

    # ── observation ──────────────────────────────────────────────────

    def _obs(self):
        field = self.game.field
        R, C = Tetris.ROWS, Tetris.COLS

        heights = np.zeros(C, dtype=np.float32)
        holes = np.zeros(C, dtype=np.float32)
        for c in range(C):
            found = False
            for r in range(R):
                if field[r][c] > 0:
                    if not found:
                        heights[c] = R - r
                        found = True
                elif found:
                    holes[c] += 1

        diffs = np.diff(heights)  # 9 values

        current = np.zeros(NUM_BLOCK_TYPES, dtype=np.float32)
        if self.game.block:
            current[self.game.block.type] = 1.0

        hold = np.zeros(NUM_BLOCK_TYPES, dtype=np.float32)
        if self.game.hold_block:
            hold[self.game.hold_block.type] = 1.0

        nxt = np.zeros(5 * NUM_BLOCK_TYPES, dtype=np.float32)
        for i, b in enumerate(list(self.game.queue)[:5]):
            nxt[i * NUM_BLOCK_TYPES + b.type] = 1.0

        obs = np.concatenate([
            heights / R, holes / R, diffs / R,
            current, hold, nxt,
        ])
        return {"obs": obs, "mask": self.action_masks()}

    # ── helpers ──────────────────────────────────────────────────────

    def _height_holes_bumpiness(self):
        """Max column height, total hole count, and total bumpiness."""
        field = self.game.field
        R, C = Tetris.ROWS, Tetris.COLS
        col_heights = np.zeros(C, dtype=int)
        max_h = 0
        total_holes = 0
        for c in range(C):
            found = False
            for r in range(R):
                if field[r][c] > 0:
                    if not found:
                        col_heights[c] = R - r
                        max_h = max(max_h, col_heights[c])
                        found = True
                elif found:
                    total_holes += 1
        bumpiness = int(np.sum(np.abs(np.diff(col_heights))))
        return max_h, total_holes, bumpiness

    def action_masks(self):
        """Boolean mask of valid placements (for masked PPO, not wired up yet)."""
        mask = np.zeros(self.action_space.n, dtype=bool)
        block = self.game.block
        if block is None:
            return mask
        orig = (block.rotation, block.x, block.y)
        for a in range(self.action_space.n):
            block.rotation = (a // Tetris.COLS) % block.num_rotations
            block.x = a % Tetris.COLS
            block.y = 0
            mask[a] = not self.game._intersects()
        block.rotation, block.x, block.y = orig
        return mask
