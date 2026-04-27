"""Root-level training entry point.

Usage:
    python train.py --algo ppo
    python train.py --algo rainbow
    python train.py --algo sac
    python train.py --algo sac_v2
    python train.py --algo sac_v2 --config rl_training/configs/sac_v2_default.json
    python train.py --algo sac_v2 --config rl_training/logs/sac_tune_winner.json --epochs 1000
    python train.py --algo dqn
    python train.py --algo all          # trains sequentially

Flags:
    --epochs N        Override max_epoch for any algo
    --config PATH     Path to JSON config file (sac_v2 only for now)
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Train Tetris RL agents")
    parser.add_argument("--algo", choices=["ppo", "rainbow", "sac", "sac_v2", "dqn", "qrdqn", "iqn", "afterstate", "all"],
                        default="all", help="Algorithm to train")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override max_epoch (default: algo-specific)")
    parser.add_argument("--config", default=None,
                        help="Path to JSON hyperparameter config (sac_v2, dqn, qrdqn, iqn)")
    args = parser.parse_args()

    algos = ["ppo", "rainbow", "sac", "sac_v2", "dqn", "qrdqn", "iqn", "afterstate"] if args.algo == "all" else [args.algo]

    for algo in algos:
        print(f"\n{'='*50}\nTraining {algo.upper()}\n{'='*50}")
        if algo == "ppo":
            from rl_training.ppo import main as run, MAX_EPOCH
            run(max_epoch=args.epochs or MAX_EPOCH)
        elif algo == "rainbow":
            from rl_training.rainbow import main as run, MAX_EPOCH
            run(max_epoch=args.epochs or MAX_EPOCH)
        elif algo == "sac":
            from rl_training.sac import main as run, MAX_EPOCH
            run(max_epoch=args.epochs or MAX_EPOCH)
        elif algo == "sac_v2":
            from rl_training.sac_v2 import main as run
            run(max_epoch=args.epochs, config_path=args.config)
        elif algo == "dqn":
            from rl_training.dqn import main as run
            run(max_epoch=args.epochs, config_path=args.config)
        elif algo == "qrdqn":
            from rl_training.qrdqn import main as run
            run(max_epoch=args.epochs, config_path=args.config)
        elif algo == "iqn":
            from rl_training.iqn import main as run
            run(max_epoch=args.epochs, config_path=args.config)
        elif algo == "afterstate":
            from rl_training.afterstate_dqn import main as run
            run(max_epoch=args.epochs, config_path=args.config)


if __name__ == "__main__":
    main()
