"""Reward shaping for Tetris RL."""

# Tetris scoring (100/300/500/800) scaled down by /100
LINE_CLEAR = [0.0, 1.0, 3.0, 5.0, 8.0]
GAME_OVER_PENALTY = -10.0
SURVIVAL_BONUS = 0.05    # small reward for each valid placement
HEIGHT_PENALTY = -0.01   # per row of max-height increase
HOLE_PENALTY = -0.03     # per new hole
BUMPINESS = -0.005       # per increase in adjacent column height difference


def compute(lines, game_over, delta_height, delta_holes, bumpiness=0):
    """Reward for a single placement."""
    r = LINE_CLEAR[min(lines, 4)] + SURVIVAL_BONUS
    if game_over:
        r += GAME_OVER_PENALTY
    r += HEIGHT_PENALTY * max(0, delta_height)
    r += HOLE_PENALTY * max(0, delta_holes)
    r += BUMPINESS * max(0, bumpiness)
    return r
