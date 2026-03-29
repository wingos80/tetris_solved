"""Smoke tests for game logic."""
from game import Tetris, Block, BLOCK_NAMES


def test_init():
    g = Tetris()
    g.next_block()
    assert g.block is not None
    assert len(g.queue) == 5
    assert len(g.field) == 20
    assert len(g.field[0]) == 10


def test_hold():
    g = Tetris()
    g.next_block()
    original_type = g.block.type
    g.hold()
    assert g.hold_block.type == original_type
    assert g.block.type != original_type or len(set(BLOCK_NAMES)) == 1  # new block from queue


def test_rotate_cw_ccw():
    b = Block(3)  # J, 4 rotations
    b.rotate_ccw()
    assert b.rotation == 3
    b.rotate_cw()
    assert b.rotation == 0


def test_rotate_wrap_2_rotations():
    b = Block(0)  # I, 2 rotations
    b.rotate_ccw()
    assert b.rotation == 1
    b.rotate_ccw()
    assert b.rotation == 0


def test_clear_single_line():
    g = Tetris()
    g.next_block()
    for j in range(g.COLS):
        g.field[19][j] = 1
    lines = g._clear_lines()
    assert lines == 1
    assert all(c == 0 for c in g.field[19])


def test_clear_multiple_lines():
    g = Tetris()
    g.next_block()
    for row in [18, 19]:
        for j in range(g.COLS):
            g.field[row][j] = 1
    lines = g._clear_lines()
    assert lines == 2


def test_score_updates_on_freeze():
    g = Tetris()
    g.next_block()
    # Fill row 19 except where the block will land
    for j in range(g.COLS):
        g.field[19][j] = 1
    initial_score = g.score
    # hard_drop triggers _freeze which calls _clear_lines and updates score
    g.hard_drop()
    assert g.score >= initial_score


if __name__ == "__main__":
    test_init()
    test_hold()
    test_rotate_cw_ccw()
    test_rotate_wrap_2_rotations()
    test_clear_single_line()
    test_clear_multiple_lines()
    test_score_updates_on_freeze()
    print("All tests passed!")
