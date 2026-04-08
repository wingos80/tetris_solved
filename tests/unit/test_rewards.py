"""Unit tests for reward computation."""

import pytest
from tetris.env.rewards import compute, LINE_CLEAR, GAME_OVER_PENALTY, SURVIVAL_BONUS


class TestLineClearReward:
    @pytest.mark.parametrize("lines,expected_base", [
        (0, 0.0),
        (1, 1.0),
        (2, 3.0),
        (3, 5.0),
        (4, 8.0),
    ])
    def test_line_clear_values(self, lines, expected_base):
        r = compute(lines, game_over=False, delta_height=0, delta_holes=0)
        assert r == pytest.approx(expected_base + SURVIVAL_BONUS)

    def test_more_than_four_clamps(self):
        r4 = compute(4, False, 0, 0)
        r5 = compute(5, False, 0, 0)
        assert r4 == r5


class TestGameOver:
    def test_game_over_penalty(self):
        r = compute(0, game_over=True, delta_height=0, delta_holes=0)
        assert r == pytest.approx(GAME_OVER_PENALTY + SURVIVAL_BONUS)

    def test_game_over_with_lines(self):
        r = compute(2, game_over=True, delta_height=0, delta_holes=0)
        assert r == pytest.approx(LINE_CLEAR[2] + GAME_OVER_PENALTY + SURVIVAL_BONUS)


class TestShaping:
    def test_height_penalty_only_positive(self):
        r_up = compute(0, False, delta_height=3, delta_holes=0)
        r_down = compute(0, False, delta_height=-2, delta_holes=0)
        assert r_up < r_down  # going up is penalized, going down is not

    def test_hole_penalty_only_positive(self):
        r_more = compute(0, False, delta_height=0, delta_holes=3)
        r_fewer = compute(0, False, delta_height=0, delta_holes=-1)
        assert r_more < r_fewer

    def test_negative_deltas_ignored(self):
        base = compute(0, False, 0, 0)
        r = compute(0, False, delta_height=-5, delta_holes=-3)
        assert r == base  # negative changes don't affect reward

    def test_survival_bonus_present(self):
        r = compute(0, False, 0, 0)
        assert r == pytest.approx(SURVIVAL_BONUS)
