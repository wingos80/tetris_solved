"""Stage 5a — MCTS planning on top of afterstate enumeration.

Wraps any afterstate V-net (e.g. afterstate_qrdqn's QuantileVNet) as a leaf
evaluator and performs a fixed-depth look-ahead via best-first expansion.

Algorithm (expectimax, not UCB):
  For each root action a:
    1. Simulate placement a on a game copy → get reward r_a, next board state.
    2. For each possible next piece (uniform over 7 types), enumerate the next
       level of afterstates and evaluate with the V-net.
    3. Q(a) = r_a + gamma * E_piece[ max_a' V(afterstate') ]
  Return argmax_a Q(a).

At depth=1 this is a 1-step expectimax (same as training, but averaged over
next-piece uncertainty). At depth>1 the recursion deepens.

Depth=1 is cheap: ~7 * ~30 = 210 V-net evals per move. Depth=2 is ~7² * 30 ≈
1470 evals, still fast given GPU batching.

Usage:
    from rl_training.mcts import mcts_action
    action = mcts_action(game, net_value_fn, depth=1, gamma=0.95)
"""

import copy

import numpy as np
import torch

from tetris.env.afterstate import apply_action, enumerate_afterstates, nuno_reward
from tetris.game import Block, Tetris

ALL_PIECE_TYPES = list(range(7))


def _clone_game(game):
    return copy.deepcopy(game)


def _expectimax(game, net_value_fn, depth, gamma):
    """Recursive expectimax over next-piece uncertainty.

    Returns a dict {action: Q_value} for the current game's active piece.
    Returns empty dict if game is over.
    """
    afterstates = enumerate_afterstates(game)
    if not afterstates:
        return {}

    actions = list(afterstates.keys())
    q_values = {}

    for action in actions:
        features, lines = afterstates[action]
        # Simulate this action on a clone to get the resulting board
        game_clone = _clone_game(game)
        actual_lines, done = apply_action(game_clone, action)
        reward = nuno_reward(actual_lines, done)

        if done or depth <= 1:
            # Leaf: use V-net on this afterstate's features
            leaf_v = float(net_value_fn(features[np.newaxis])[0])
            q_values[action] = reward + gamma * leaf_v
        else:
            # Recurse: average over 7 possible next pieces
            next_vals = []
            for piece_type in ALL_PIECE_TYPES:
                g_next = _clone_game(game_clone)
                # Override the spawned block with this hypothetical next piece
                g_next.block = Block(piece_type)
                g_next.block.x = Tetris.COLS // 2 - 2
                g_next.block.y = 0
                if g_next._intersects():
                    # Spawn blocked → board is full, terminal
                    next_vals.append(0.0)
                    continue
                child_qs = _expectimax(g_next, net_value_fn, depth - 1, gamma)
                if not child_qs:
                    next_vals.append(0.0)
                else:
                    next_vals.append(max(child_qs.values()))
            q_values[action] = reward + gamma * float(np.mean(next_vals))

    return q_values


def mcts_action(game, net_value_fn, depth=1, gamma=0.95):
    """Choose the best action for the current game state using expectimax lookahead.

    Args:
        game: Tetris game instance (not mutated).
        net_value_fn: callable(features_array) -> values_array.
                      Input: [N, 4] float32; output: [N] float.
                      Compatible with VNet, QuantileVNet.value, etc.
        depth: lookahead depth. 1 = next-piece expectation only (cheap).
               2 = two-ply (7× more expensive).
        gamma: discount applied at each depth level.

    Returns:
        (action, features) where action = (rot, x) tuple, or (None, None) if game over.
    """
    afterstates = enumerate_afterstates(game)
    if not afterstates:
        return None, None

    with torch.no_grad():
        q_values = _expectimax(game, net_value_fn, depth, gamma)
    if not q_values:
        return None, None

    best_action = max(q_values, key=lambda a: q_values[a])
    return best_action, afterstates[best_action][0]
