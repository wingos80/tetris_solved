"""Afterstate enumeration for value-based RL (nuno-faria style).

For a given Tetris game state, enumerates every valid (rotation, x) placement,
simulates the drop + line clear, and returns the resulting 4-feature vector
[lines_cleared, holes, bumpiness, total_height].

Action key is the tuple (rotation, x) where x is block.x. x ranges over
[-3, COLS-1] to cover placements where the block's bbox extends past the
left edge (cells at col 0 of the bbox put no actual cells off the board).
"""

import numpy as np

from tetris.game import Tetris


def state_props(field, lines_cleared):
    """Compute [lines_cleared, holes, bumpiness, total_height] for a field."""
    R, C = Tetris.ROWS, Tetris.COLS
    heights = [0] * C
    holes = 0
    for c in range(C):
        seen = False
        for r in range(R):
            if field[r][c] > 0:
                if not seen:
                    heights[c] = R - r
                    seen = True
            elif seen:
                holes += 1
    bumpiness = sum(abs(heights[i + 1] - heights[i]) for i in range(C - 1))
    total_height = sum(heights)
    return np.array([lines_cleared, holes, bumpiness, total_height], dtype=np.float32)


def _simulate_placement(game, rot, x):
    """Return (field_after, lines_cleared, would_be_gameover) for placement.

    Returns None if placement is invalid (block intersects at spawn position).
    Does NOT mutate game.
    """
    block = game.block
    R, C = Tetris.ROWS, Tetris.COLS
    orig = (block.rotation, block.x, block.y)
    block.rotation, block.x, block.y = rot, x, 0
    try:
        if game._intersects():
            return None
        # Drop
        while not game._intersects():
            block.y += 1
        block.y -= 1
        # Snapshot field, freeze block in copy
        field_copy = [row[:] for row in game.field]
        for dr, dc in block.cells:
            rr, cc = dr + block.y, dc + block.x
            if 0 <= rr < R and 0 <= cc < C:
                field_copy[rr][cc] = block.type + 1
        # Clear lines on copy
        lines = 0
        for i in range(R):
            if all(cell > 0 for cell in field_copy[i]):
                lines += 1
                for k in range(i, 0, -1):
                    field_copy[k] = field_copy[k - 1][:]
                field_copy[0] = [0] * C
        # Game over heuristic: any block above row 0 — but our game flags gameover
        # only when next spawn collides. We can't know that without spawning, so
        # leave done=False here; the training loop checks game.state after applying.
        return field_copy, lines, False
    finally:
        block.rotation, block.x, block.y = orig


def enumerate_afterstates(game):
    """Enumerate every valid placement across the full x range.

    Returns: dict {(rotation, x): (features_4d, lines_cleared)}
    where x can be negative (block bbox extends past left edge but cells stay on board).
    """
    out = {}
    block = game.block
    if block is None or game.state == "gameover":
        return out
    C = Tetris.COLS
    # Block cells live in a 4x4 bbox; x can range from -3 to COLS-1
    # (placements with all cells out-of-bounds are filtered by _intersects).
    for rot in range(block.num_rotations):
        for x in range(-3, C):
            sim = _simulate_placement(game, rot, x)
            if sim is None:
                continue
            field_after, lines, _ = sim
            features = state_props(field_after, lines)
            out[(rot, x)] = (features, lines)
    return out


def apply_action(game, action):
    """Apply a placement action (rot, x) tuple to the real game.

    Returns (lines, gameover). Skips reward shaping — caller computes reward.
    """
    rot, x = action
    block = game.block
    block.rotation = rot % block.num_rotations
    block.x = x
    block.y = 0
    if game._intersects():
        return 0, True  # invalid -> treat as terminal
    lines = game.hard_drop()
    return lines, game.state == "gameover"


def nuno_reward(lines, gameover, board_width=Tetris.COLS):
    """Reward formula from nuno-faria/tetris-ai: 1 + lines^2 * width - 2 if gameover."""
    r = 1.0 + (lines ** 2) * board_width
    if gameover:
        r -= 2.0
    return r
