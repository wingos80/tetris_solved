"""Train a Double DQN agent on Tetris using tianshou (0.5.x).

3-hidden-layer MLP, epsilon-greedy, native action masking via dict obs.
Hyperparameters loaded from JSON config — see configs/dqn_default.json.

Usage:
    python train.py --algo dqn
    python train.py --algo dqn --config rl_training/configs/dqn_default.json
    python train.py --algo dqn --config rl_training/logs/dqn_tune_winner.json
"""

import json
import torch
import torch.nn as nn
from pathlib import Path

WEIGHTS_DIR = Path(__file__).parent / "weights"
DEFAULT_CONFIG = Path(__file__).parent / "configs" / "dqn_default.json"

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import DQNPolicy
from tianshou.trainer import offpolicy_trainer

from tetris.env import TetrisEnv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_TRAIN = 8
NUM_TEST = 4


# ── Q-network ─────────────────────────────────────────────────────

class QNet(nn.Module):
    """MLP outputting Q-values for all actions.

    Handles both plain tensor obs and tianshou dict obs (obs.obs).
    """

    def __init__(self, obs_n, act_n, hidden, device):
        super().__init__()
        layers = []
        in_dim = obs_n
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, act_n))
        self.net = nn.Sequential(*layers).to(device)

    def forward(self, obs, state=None, info={}):
        if hasattr(obs, "obs"):
            obs = obs.obs
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32,
                                  device=next(self.parameters()).device)
        return self.net(obs), state


# ── config ────────────────────────────────────────────────────────

def load_config(path=None):
    """Load config from dqn_default.json, with optional override from path."""
    with open(DEFAULT_CONFIG) as f:
        cfg = json.load(f)
    if path is not None:
        with open(path) as f:
            cfg.update(json.load(f))
    return cfg


# ── trial runner (used by tune.py) ───────────────────────────────

def run_trial(cfg: dict, trial_epochs: int) -> float:
    """Short training trial for hyperparameter search.

    No weights saved, no tensorboard. Returns best test_reward seen.
    """
    obs_n, act_n = 78, 40

    net = QNet(obs_n, act_n, cfg["hidden"], DEVICE)
    optim = torch.optim.Adam(net.parameters(), lr=cfg["lr"])

    policy = DQNPolicy(
        net, optim,
        discount_factor=cfg["gamma"],
        estimation_step=int(cfg["n_step"]),
        target_update_freq=int(cfg["target_update_freq"]),
        is_double=True,
    ).to(DEVICE)
    policy.set_eps(cfg["eps_train"])

    train_envs = DummyVectorEnv([lambda: TetrisEnv() for _ in range(NUM_TRAIN)])
    test_envs = DummyVectorEnv([lambda: TetrisEnv() for _ in range(NUM_TEST)])

    train_collector = Collector(
        policy, train_envs,
        VectorReplayBuffer(int(cfg["buffer_size"]), NUM_TRAIN),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs, exploration_noise=False)

    reset_freq = int(cfg.get("reset_freq", 0))

    def train_fn(epoch, env_step):
        policy.set_eps(cfg["eps_train"])
        if reset_freq and epoch % reset_freq == 0:
            net.net[-1].reset_parameters()

    def test_fn(epoch, env_step): policy.set_eps(cfg["eps_test"])

    result = offpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=trial_epochs,
        step_per_epoch=int(cfg["step_per_epoch"]),
        step_per_collect=int(cfg["step_per_collect"]),
        update_per_step=cfg["update_per_step"],
        episode_per_test=int(cfg["episode_per_test"]),
        batch_size=int(cfg["batch_size"]),
        train_fn=train_fn,
        test_fn=test_fn,
        save_best_fn=lambda p: None,
        save_checkpoint_fn=lambda e, es, gs: None,
        verbose=False,
    )

    train_envs.close()
    test_envs.close()
    return float(result.get("best_reward", result.get("rew", 0.0)))


# ── full training run ─────────────────────────────────────────────

def main(max_epoch=None, config_path=None):
    cfg = load_config(config_path)
    if max_epoch is not None:
        cfg["max_epoch"] = max_epoch

    print(f"DQN config: {json.dumps(cfg, indent=2)}")

    from torch.utils.tensorboard import SummaryWriter
    from tianshou.utils import TensorboardLogger
    log_dir = WEIGHTS_DIR.parent / "logs" / "dqn"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = TensorboardLogger(SummaryWriter(str(log_dir)))

    train_envs = DummyVectorEnv([lambda: TetrisEnv() for _ in range(NUM_TRAIN)])
    test_envs = DummyVectorEnv([lambda: TetrisEnv() for _ in range(NUM_TEST)])

    obs_n, act_n = 78, 40

    net = QNet(obs_n, act_n, cfg["hidden"], DEVICE)
    optim = torch.optim.Adam(net.parameters(), lr=cfg["lr"])

    policy = DQNPolicy(
        net, optim,
        discount_factor=cfg["gamma"],
        estimation_step=int(cfg["n_step"]),
        target_update_freq=int(cfg["target_update_freq"]),
        is_double=True,
    ).to(DEVICE)

    train_collector = Collector(
        policy, train_envs,
        VectorReplayBuffer(int(cfg["buffer_size"]), NUM_TRAIN),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs, exploration_noise=False)

    def save_best(policy):
        WEIGHTS_DIR.mkdir(exist_ok=True)
        torch.save(policy.state_dict(), WEIGHTS_DIR / "best_dqn.pth")
        print(f"  -> saved rl_training/weights/best_dqn.pth")

    ckpt_dir = WEIGHTS_DIR / "checkpoints"
    checkpoint_freq = int(cfg["checkpoint_freq"])

    def save_checkpoint(epoch, env_step, gradient_step):
        if epoch % checkpoint_freq == 0:
            ckpt_dir.mkdir(exist_ok=True)
            path = ckpt_dir / f"dqn_ep{epoch:04d}.pth"
            torch.save(policy.state_dict(), path)
            print(f"  -> checkpoint: {path.name}")

    reset_freq = int(cfg.get("reset_freq", 0))

    def train_fn(epoch, env_step):
        policy.set_eps(cfg["eps_train"])
        if reset_freq and epoch % reset_freq == 0:
            net.net[-1].reset_parameters()
            print(f"  -> reset output head (epoch {epoch})")

    def test_fn(epoch, env_step): policy.set_eps(cfg["eps_test"])

    result = offpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=int(cfg["max_epoch"]),
        step_per_epoch=int(cfg["step_per_epoch"]),
        step_per_collect=int(cfg["step_per_collect"]),
        update_per_step=cfg["update_per_step"],
        episode_per_test=int(cfg["episode_per_test"]),
        batch_size=int(cfg["batch_size"]),
        train_fn=train_fn,
        test_fn=test_fn,
        save_best_fn=save_best,
        save_checkpoint_fn=save_checkpoint,
        logger=logger,
        verbose=True,
    )

    print(f"\nTraining complete. Result: {result}")


if __name__ == "__main__":
    main()
