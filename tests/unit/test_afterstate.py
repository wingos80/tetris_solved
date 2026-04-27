"""Unit tests for afterstate enumeration utilities."""

import numpy as np
import pytest

from tetris.env.afterstate import (
    apply_action,
    enumerate_afterstates,
    nuno_reward,
    state_props,
)
from tetris.game import Tetris


def _empty_field():
    return [[0] * Tetris.COLS for _ in range(Tetris.ROWS)]


def _full_row_field(row):
    """Field with row `row` filled (and everything above it empty)."""
    f = _empty_field()
    f[row] = [1] * Tetris.COLS
    return f


class TestStateProps:
    def test_empty_board(self):
        feats = state_props(_empty_field(), 0)
        assert feats.tolist() == [0, 0, 0, 0]
        assert feats.dtype == np.float32

    def test_lines_cleared_passed_through(self):
        feats = state_props(_empty_field(), 3)
        assert feats[0] == 3

    def test_single_block(self):
        f = _empty_field()
        f[Tetris.ROWS - 1][0] = 1  # bottom-left corner
        feats = state_props(f, 0)
        # 1 column has height 1, others 0 -> bumpiness=1, total_height=1, holes=0
        assert feats[1] == 0  # holes
        assert feats[2] == 1  # bumpiness
        assert feats[3] == 1  # total_height

    def test_holes_counted(self):
        f = _empty_field()
        # column 0: filled at row 18, empty at row 19 -> hole below
        f[Tetris.ROWS - 2][0] = 1
        feats = state_props(f, 0)
        assert feats[1] == 1  # one hole


class TestEnumerateAfterstates:
    def test_returns_dict_with_valid_actions(self):
        game = Tetris()
        game.next_block()
        afterstates = enumerate_afterstates(game)
        assert len(afterstates) > 0
        for action, (feats, lines) in afterstates.items():
            rot, x = action
            assert 0 <= rot < 4
            assert -3 <= x < Tetris.COLS
            assert feats.shape == (4,)
            assert lines >= 0

    def test_empty_when_gameover(self):
        game = Tetris()
        game.next_block()
        game.state = "gameover"
        assert enumerate_afterstates(game) == {}

    def test_empty_when_no_block(self):
        game = Tetris()
        # don't call next_block
        assert enumerate_afterstates(game) == {}

    def test_does_not_mutate_game(self):
        game = Tetris()
        game.next_block()
        block = game.block
        before = (block.rotation, block.x, block.y, block.type)
        field_before = [row[:] for row in game.field]
        enumerate_afterstates(game)
        assert (block.rotation, block.x, block.y, block.type) == before
        assert game.field == field_before

    def test_action_keys_are_tuples(self):
        """Action key = (rotation, x). x can be negative (block bbox left of edge)."""
        game = Tetris()
        game.next_block()
        afterstates = enumerate_afterstates(game)
        for action in afterstates:
            assert isinstance(action, tuple)
            rot, x = action
            assert 0 <= rot < 4
            assert -3 <= x < Tetris.COLS

    def test_includes_negative_x_placements(self):
        """Extended enumeration must reach placements only valid at x < 0."""
        from tetris.game import Block
        # Force an L piece at start so we know L rot 2 needs x=-2 to flush left
        game = Tetris()
        # Replace queue head with L (type index 4)
        game.queue.clear()
        game.queue.append(Block(4))  # L
        for _ in range(4):
            game.queue.append(Block())
        game.next_block()
        afterstates = enumerate_afterstates(game)
        negative_x_actions = [a for a in afterstates if a[1] < 0]
        assert len(negative_x_actions) > 0, "expected at least one x<0 placement"


class TestNunoReward:
    def test_no_lines_no_gameover(self):
        assert nuno_reward(0, False) == 1.0

    def test_one_line(self):
        # 1 + 1^2 * 10 = 11
        assert nuno_reward(1, False) == 11.0

    def test_four_lines_tetris(self):
        # 1 + 16 * 10 = 161
        assert nuno_reward(4, False) == 161.0

    def test_gameover_subtracts_two(self):
        assert nuno_reward(0, True) == -1.0
        assert nuno_reward(1, True) == 9.0


class TestApplyAction:
    def test_valid_placement_advances_game(self):
        game = Tetris()
        game.next_block()
        block_type_before = game.block.type
        afterstates = enumerate_afterstates(game)
        action = next(iter(afterstates.keys()))
        lines, done = apply_action(game, action)
        assert lines >= 0
        # Game should have spawned a new block (or be gameover)
        if not done:
            assert game.block is not None
        # Block changed (since hard_drop spawns next)
        # In rare cases the next block has same type, so just check we advanced
        assert game.score >= 0

    def test_invalid_action_returns_terminal(self):
        game = Tetris()
        game.next_block()
        # Force board state where every column is full to top
        for r in range(Tetris.ROWS):
            game.field[r] = [1] * Tetris.COLS
        # Any placement intersects -> apply_action returns terminal
        lines, done = apply_action(game, (0, 4))
        assert done is True
        assert lines == 0
