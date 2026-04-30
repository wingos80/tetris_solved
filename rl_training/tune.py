"""Generic Monte Carlo hyperparameter search for Tetris RL agents.

Each algo module must expose:
    run_trial(cfg: dict, trial_epochs: int) -> float

The search space is a JSON file with a "space" key:
    {
      "space": {
        "lr": ["loguniform", 1e-5, 1e-3],
        "gamma": ["uniform", 0.97, 0.999],
        "n_step": ["choice", [3, 5, 7]],
        "hidden": ["choice", [[256,256],[256,256,256]]]
      }
    }

Distribution types:
    ["loguniform", lo, hi]   -- exp(uniform(log(lo), log(hi))), float
    ["uniform", lo, hi]      -- uniform float
    ["int_uniform", lo, hi]  -- uniform int in [lo, hi] inclusive
    ["choice", [v1, v2, ...]] -- one element, type preserved

Only keys listed in "space" are sampled. All other parameters come from
the algo's default config (rl_training/configs/{algo}_default.json).
The winner JSON is the full merged config, ready for train.py --config.

Usage:
    python rl_training/tune.py --algo sac_v2
    python rl_training/tune.py --algo dqn --trials 20 --epochs 30
    python rl_training/tune.py --algo sac_v2 --space my_space.json
    python rl_training/tune.py --algo sac_v2 --out results/my_winner.json
"""

import argparse
import importlib
import json
import math
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent

# ── algo registry ─────────────────────────────────────────────────
# Add new algos here (+ default config + search space + run_trial in module)
ALGO_MODULES = {
    "sac_v2":  "rl_training.sac_v2",
    "dqn":     "rl_training.dqn",
}

DEFAULT_CONFIG_PATHS = {
    "sac_v2":  ROOT / "rl_training" / "configs" / "sac_v2_default.json",
    "dqn":     ROOT / "rl_training" / "configs" / "dqn_default.json",
}

DEFAULT_SPACE_PATHS = {
    "sac_v2":  ROOT / "rl_training" / "configs" / "sac_v2_search.json",
    "dqn":     ROOT / "rl_training" / "configs" / "dqn_search.json",
}


# ── sampler ───────────────────────────────────────────────────────

def _sample_one(spec, rng, nprng):
    kind = spec[0]
    if kind == "loguniform":
        lo, hi = math.log(spec[1]), math.log(spec[2])
        return float(math.exp(nprng.uniform(lo, hi)))
    elif kind == "uniform":
        return float(nprng.uniform(spec[1], spec[2]))
    elif kind == "int_uniform":
        return int(nprng.randint(spec[1], spec[2] + 1))
    elif kind == "choice":
        return rng.choice(spec[1])
    else:
        raise ValueError(f"Unknown distribution type: {kind!r}")


def sample_config(space: dict, seed: int) -> dict:
    rng = random.Random(seed)
    nprng = np.random.RandomState(seed)
    return {k: _sample_one(v, rng, nprng) for k, v in space.items()}


# ── trial runner ──────────────────────────────────────────────────

def run_trial(module, cfg: dict, trial_epochs: int) -> float:
    return module.run_trial(cfg, trial_epochs)


# ── result formatting ─────────────────────────────────────────────

def _cfg_summary(cfg: dict, space_keys: list) -> str:
    """Short string of just the searched keys."""
    parts = []
    for k in space_keys:
        v = cfg[k]
        if isinstance(v, float):
            parts.append(f"{k}={v:.3g}")
        else:
            parts.append(f"{k}={v}")
    return "  ".join(parts)


# ── main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Monte Carlo hyperparameter search for Tetris RL agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--algo", choices=list(ALGO_MODULES), required=True,
                        help="Algorithm to tune")
    parser.add_argument("--space", default=None,
                        help="Path to search space JSON (default: rl_training/configs/{algo}_search.json)")
    parser.add_argument("--trials", type=int, default=12,
                        help="Number of MC trials (default: 12)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Epochs per trial (default: 50)")
    parser.add_argument("--out", default=None,
                        help="Path to write winner config JSON "
                             "(default: rl_training/logs/{algo}_tune_winner.json)")
    args = parser.parse_args()

    # ── load space ────────────────────────────────────────────────
    space_path = Path(args.space) if args.space else DEFAULT_SPACE_PATHS[args.algo]
    with open(space_path) as f:
        space_doc = json.load(f)
    space = space_doc["space"]

    # ── load base config ─────────────────────────────────────────
    with open(DEFAULT_CONFIG_PATHS[args.algo]) as f:
        base_cfg = json.load(f)

    # ── import algo module ────────────────────────────────────────
    module = importlib.import_module(ALGO_MODULES[args.algo])
    if not hasattr(module, "run_trial"):
        print(f"[error] {ALGO_MODULES[args.algo]} does not expose run_trial(cfg, trial_epochs)")
        sys.exit(1)

    # ── log dir ───────────────────────────────────────────────────
    log_dir = ROOT / "rl_training" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    results_path = log_dir / f"{args.algo}_tune_results.json"
    winner_path = Path(args.out) if args.out else log_dir / f"{args.algo}_tune_winner.json"

    print(f"Tuning {args.algo.upper()} — {args.trials} trials × {args.epochs} epochs each")
    print(f"Search space ({len(space)} params): {', '.join(space)}")
    print(f"Base config: {DEFAULT_CONFIG_PATHS[args.algo]}")
    print()

    results = []

    for i in range(args.trials):
        sampled = sample_config(space, seed=i)
        cfg = {**base_cfg, **sampled}

        print(f"[Trial {i+1}/{args.trials}]  seed={i}")
        print(f"  {_cfg_summary(sampled, list(space))}")

        t0 = time.time()
        try:
            best_rew = run_trial(module, cfg, args.epochs)
        except Exception:
            print(f"  [ERROR] trial failed:")
            traceback.print_exc()
            best_rew = float("-inf")
        elapsed = time.time() - t0

        print(f"  best_reward={best_rew:.3f}  ({elapsed:.0f}s)")
        results.append({"seed": i, "best_reward": best_rew, "config": cfg})

    # ── sort and report ───────────────────────────────────────────
    results.sort(key=lambda x: x["best_reward"], reverse=True)

    print(f"\n{'='*60}")
    print(f"RESULTS — {args.algo.upper()} (sorted by best_reward):")
    print(f"{'Rank':>4}  {'Reward':>8}  {'Seed':>4}  Config")
    for rank, r in enumerate(results, 1):
        summary = _cfg_summary(r["config"], list(space))
        print(f"{rank:>4}  {r['best_reward']:>8.3f}  {r['seed']:>4}  {summary}")

    winner = results[0]
    print(f"\nWinner (seed={winner['seed']}, reward={winner['best_reward']:.3f}):")
    for k, v in winner["config"].items():
        marker = " *" if k in space else ""
        print(f"  {k} = {v}{marker}")

    # ── save results ──────────────────────────────────────────────
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results: {results_path}")

    with open(winner_path, "w") as f:
        json.dump(winner["config"], f, indent=2)
    print(f"Winner config: {winner_path}")
    print(f"\nTo train with winner config:")
    print(f"  python train.py --algo {args.algo} --config {winner_path}")


if __name__ == "__main__":
    main()
