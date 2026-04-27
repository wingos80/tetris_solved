"""Standardized eval metrics for any agent — reward / lines / placements.

Runs N episodes greedily on saved weights and reports mean ± std for each
metric. Use this to compare agents objectively across algorithm and reward
shaping changes (raw reward numbers aren't comparable; lines and placements
are).

Usage:
    python eval_metrics.py --algo afterstate
    python eval_metrics.py --algo sac_v2 --episodes 30
    python eval_metrics.py --algo dqn path/to/best.pth
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "rl_training"))

from tetris.env import TetrisEnv
from tetris.env.afterstate import apply_action, enumerate_afterstates, nuno_reward
from tetris.game import Tetris

from eval import BUILDERS, _PATH_AWARE_BUILDERS, get_action

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _latest_weights(algo):
    algo_dir = ROOT / "rl_training" / "weights" / algo
    if not algo_dir.is_dir():
        return None
    runs = sorted(
        [d for d in algo_dir.iterdir() if d.is_dir() and (d / "best.pth").exists()],
        key=lambda d: d.name, reverse=True,
    )
    return str(runs[0] / "best.pth") if runs else None


def eval_tianshou_agent(path, algo, episodes):
    """Eval a tianshou-policy agent. Returns lists of (rewards, lines, placements)."""
    if algo in _PATH_AWARE_BUILDERS:
        policy = BUILDERS[algo](weights_path=path)
    else:
        policy = BUILDERS[algo]()
    policy.load_state_dict(torch.load(path, map_location=DEVICE))
    policy.to(DEVICE).eval()

    rewards, lines_list, placements = [], [], []
    for _ in range(episodes):
        env = TetrisEnv()
        obs, _ = env.reset()
        ep_reward, ep_lines, ep_placements = 0.0, 0, 0
        done = False
        while not done:
            action = get_action(policy, obs, algo)
            obs, r, term, trunc, info = env.step(action)
            ep_reward += r
            if info.get("valid", False):
                ep_placements += 1
                ep_lines += info.get("lines", 0)
            done = term or trunc
            if ep_placements > 5000:  # safety cap
                break
        rewards.append(ep_reward)
        lines_list.append(ep_lines)
        placements.append(ep_placements)
    return rewards, lines_list, placements


def eval_afterstate_agent(path, episodes):
    """Eval afterstate agent. Reward computed via nuno formula."""
    from rl_training.afterstate_dqn import VNet
    import json
    cfg_path = Path(path).parent / "config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    net = VNet(cfg["hidden"], DEVICE)
    net.load_state_dict(torch.load(path, map_location=DEVICE))
    net.eval()

    rewards, lines_list, placements = [], [], []
    for _ in range(episodes):
        game = Tetris()
        game.next_block()
        ep_reward, ep_lines, ep_placements = 0.0, 0, 0
        while game.state != "gameover":
            afterstates = enumerate_afterstates(game)
            if not afterstates:
                break
            actions = list(afterstates.keys())
            feats = np.stack([afterstates[a][0] for a in actions])
            with torch.no_grad():
                values = net(feats).cpu().numpy()
            action = actions[int(np.argmax(values))]
            lines, done = apply_action(game, action)
            ep_reward += nuno_reward(lines, done)
            ep_lines += lines
            ep_placements += 1
            if done or ep_placements > 5000:
                break
        rewards.append(ep_reward)
        lines_list.append(ep_lines)
        placements.append(ep_placements)
    return rewards, lines_list, placements


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default=None)
    parser.add_argument("--algo", required=True,
                        choices=["ppo", "rainbow", "sac", "sac_v2", "dqn",
                                 "qrdqn", "iqn", "afterstate"])
    parser.add_argument("--episodes", type=int, default=30)
    args = parser.parse_args()

    if args.path is None:
        args.path = _latest_weights(args.algo)
        if args.path is None:
            print(f"No weights found in weights/{args.algo}/")
            sys.exit(1)
        print(f"Using latest weights: {args.path}")

    print(f"Evaluating {args.algo} for {args.episodes} episodes...")
    if args.algo == "afterstate":
        rewards, lines, placements = eval_afterstate_agent(args.path, args.episodes)
    else:
        rewards, lines, placements = eval_tianshou_agent(args.path, args.algo, args.episodes)

    rewards = np.array(rewards)
    lines = np.array(lines)
    placements = np.array(placements)
    print(f"\n{'='*50}")
    print(f"Algo: {args.algo}  |  Weights: {args.path}")
    print(f"Episodes: {args.episodes}")
    print(f"{'='*50}")
    print(f"Reward:     {rewards.mean():>8.2f} ± {rewards.std():.2f}   "
          f"[min {rewards.min():.2f}, max {rewards.max():.2f}]")
    print(f"Lines:      {lines.mean():>8.2f} ± {lines.std():.2f}   "
          f"[min {lines.min()}, max {lines.max()}]")
    print(f"Placements: {placements.mean():>8.2f} ± {placements.std():.2f}   "
          f"[min {placements.min()}, max {placements.max()}]")


if __name__ == "__main__":
    main()
