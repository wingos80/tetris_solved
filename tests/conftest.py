"""Shared fixtures for the test suite."""

import pytest
from tetris import Tetris, Block


@pytest.fixture
def game():
    """A fresh Tetris game with the first block spawned."""
    g = Tetris()
    g.next_block()
    return g


@pytest.fixture
def empty_game():
    """A Tetris game with no active block (pre-spawn)."""
    return Tetris()


@pytest.fixture
def full_row_game(game):
    """A game with row 19 (bottom) completely filled."""
    for j in range(Tetris.COLS):
        game.field[19][j] = 1
    return game
