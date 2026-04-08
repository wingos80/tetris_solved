"""Unit tests for the Block class."""

import pytest
from tetris import Block, BLOCK_SHAPES, BLOCK_NAMES, NUM_BLOCK_TYPES


class TestBlockInit:
    def test_random_type(self):
        types = {Block().type for _ in range(200)}
        assert len(types) > 1, "Random blocks should produce multiple types"

    def test_explicit_type(self):
        for i in range(NUM_BLOCK_TYPES):
            assert Block(i).type == i

    def test_initial_rotation_zero(self):
        assert Block(0).rotation == 0

    def test_initial_position_zero(self):
        b = Block(0)
        assert (b.x, b.y) == (0, 0)


class TestBlockRotation:
    @pytest.mark.parametrize("type_idx,expected_rots", [
        (0, 2),   # I
        (1, 2),   # Z
        (2, 2),   # S
        (3, 4),   # J
        (4, 4),   # L
        (5, 4),   # T
        (6, 1),   # O
    ])
    def test_num_rotations(self, type_idx, expected_rots):
        assert Block(type_idx).num_rotations == expected_rots

    def test_cw_wraps(self):
        b = Block(0)  # I: 2 rotations
        b.rotate_cw()
        assert b.rotation == 1
        b.rotate_cw()
        assert b.rotation == 0

    def test_ccw_wraps(self):
        b = Block(0)  # I: 2 rotations
        b.rotate_ccw()
        assert b.rotation == 1
        b.rotate_ccw()
        assert b.rotation == 0

    def test_cw_ccw_inverse(self):
        b = Block(3)  # J: 4 rotations
        b.rotate_cw()
        b.rotate_ccw()
        assert b.rotation == 0

    def test_ccw_from_zero(self):
        b = Block(3)  # J: 4 rotations
        b.rotate_ccw()
        assert b.rotation == 3

    def test_full_cw_cycle(self):
        for t in range(NUM_BLOCK_TYPES):
            b = Block(t)
            for _ in range(b.num_rotations):
                b.rotate_cw()
            assert b.rotation == 0, f"{BLOCK_NAMES[t]} didn't cycle back"


class TestBlockCells:
    def test_cells_returns_four_tuples(self):
        for t in range(NUM_BLOCK_TYPES):
            b = Block(t)
            assert len(b.cells) == 4
            for r, c in b.cells:
                assert 0 <= r < 4 and 0 <= c < 4

    def test_cells_change_with_rotation(self):
        b = Block(0)  # I: definitely different rotations
        cells_0 = b.cells
        b.rotate_cw()
        cells_1 = b.cells
        assert cells_0 != cells_1

    def test_o_block_single_rotation(self):
        b = Block(6)  # O: 1 rotation
        assert b.num_rotations == 1
        cells = b.cells
        b.rotate_cw()  # should wrap to 0
        assert b.cells == cells
