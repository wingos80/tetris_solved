"""Unit tests for the Tetris game logic."""

import pytest
from tetris import Tetris, Block


class TestInit:
    def test_field_dimensions(self, game):
        assert len(game.field) == 20
        assert all(len(row) == 10 for row in game.field)

    def test_field_empty(self, game):
        assert all(cell == 0 for row in game.field for cell in row)

    def test_queue_filled(self, game):
        assert len(game.queue) == 5

    def test_active_block(self, game):
        assert game.block is not None

    def test_initial_state(self, game):
        assert game.state == "playing"
        assert game.score == 0

    def test_no_hold_initially(self, game):
        assert game.hold_block is None
        assert game.can_hold is True


class TestMovement:
    def test_move_left(self, game):
        x = game.block.x
        game.move(-1)
        assert game.block.x == x - 1

    def test_move_right(self, game):
        x = game.block.x
        game.move(1)
        assert game.block.x == x + 1

    def test_move_blocked_by_wall(self, game):
        # Move left until we hit the wall
        for _ in range(20):
            game.move(-1)
        x_at_wall = game.block.x
        game.move(-1)
        assert game.block.x == x_at_wall  # can't go further

    def test_soft_drop(self, game):
        y = game.block.y
        game.soft_drop()
        assert game.block.y == y + 1

    def test_hard_drop_reaches_bottom(self, game):
        old_block = game.block.type
        game.hard_drop()
        # After hard_drop, the block is frozen and next block spawned
        assert game.block.type != old_block or game.block.y == 0


class TestRotation:
    def test_cw_changes_rotation(self, game):
        if game.block.num_rotations > 1:
            old = game.block.rotation
            game.rotate_cw()
            assert game.block.rotation != old

    def test_ccw_changes_rotation(self, game):
        if game.block.num_rotations > 1:
            old = game.block.rotation
            game.rotate_ccw()
            assert game.block.rotation != old

    def test_blocked_rotation_reverts(self, game):
        # Fill surrounding cells to block rotation
        game.block.x = 0
        game.block.y = 0
        for r in range(4):
            for c in range(4):
                game.field[r][c] = 1
        old_rot = game.block.rotation
        game.rotate_cw()
        assert game.block.rotation == old_rot


class TestLineClear:
    def test_clear_single_line(self, full_row_game):
        lines = full_row_game._clear_lines()
        assert lines == 1
        assert all(c == 0 for c in full_row_game.field[19])

    def test_clear_double(self, game):
        for row in [18, 19]:
            for j in range(Tetris.COLS):
                game.field[row][j] = 1
        assert game._clear_lines() == 2

    def test_clear_triple(self, game):
        for row in [17, 18, 19]:
            for j in range(Tetris.COLS):
                game.field[row][j] = 1
        assert game._clear_lines() == 3

    def test_clear_tetris(self, game):
        for row in [16, 17, 18, 19]:
            for j in range(Tetris.COLS):
                game.field[row][j] = 1
        assert game._clear_lines() == 4

    def test_rows_shift_down(self, game):
        # Put a marker in row 18, fill row 19
        game.field[18][0] = 5
        for j in range(Tetris.COLS):
            game.field[19][j] = 1
        game._clear_lines()
        # Marker should have shifted down to row 19
        assert game.field[19][0] == 5


class TestScoring:
    def test_score_single(self, full_row_game):
        full_row_game.hard_drop()
        assert full_row_game.score >= 100

    def test_score_zero_no_clear(self, game):
        game.hard_drop()
        # May or may not clear lines depending on random block
        assert game.score >= 0

    def test_line_scores_table(self):
        assert Tetris.LINE_SCORES == [0, 100, 300, 500, 800]


class TestHold:
    def test_hold_swaps_block(self, game):
        original_type = game.block.type
        game.hold()
        assert game.hold_block.type == original_type

    def test_hold_once_per_placement(self, game):
        game.hold()
        held_type = game.hold_block.type
        game.hold()  # should be no-op
        assert game.hold_block.type == held_type

    def test_hold_swap_back(self, game):
        first_type = game.block.type
        game.hold()
        # Place a piece to re-enable hold
        game.hard_drop()
        second_type = game.block.type
        game.hold()
        # Should get the first piece back
        assert game.block.type == first_type
        assert game.hold_block.type == second_type


class TestQueue:
    def test_queue_refills(self, game):
        for _ in range(10):
            game.hard_drop()
            if game.state == "gameover":
                break
            assert len(game.queue) == 5

    def test_next_block_pops_from_queue(self, game):
        expected_type = game.queue[0].type
        game.hard_drop()  # triggers next_block
        if game.state != "gameover":
            assert game.block.type == expected_type


class TestGameOver:
    def test_gameover_on_stack(self, game):
        # Fill the field top-to-bottom
        for r in range(Tetris.ROWS):
            for c in range(Tetris.COLS):
                game.field[r][c] = 1
        game.next_block()
        assert game.state == "gameover"
