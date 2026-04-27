"""Discrete SAC (v2) for Tetris — config-driven.

Hyperparameters are loaded from a JSON config file so they can be
tuned externally (e.g. via tune_sac.py) and passed in without touching
this file.

Default config: rl_training/configs/sac_v2_default.json

Usage:
    python train.py --algo sac_v2
    python train.py --algo sac_v2 --config rl_training/configs/sac_v2_default.json
    python train.py --algo sac_v2 --config rl_training/logs/sac_tune_winner.json --epochs 1000
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

WEIGHTS_DIR = Path(__file__).parent / "weights"
DEFAULT_CONFIG = Path(__file__).parent / "configs" / "sac_v2_default.json"

from tianshou.data import Collector, PrioritizedVectorReplayBuffer, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import DiscreteSACPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.common import Net

from tetris.env import TetrisEnv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_TRAIN = 8
NUM_TEST = 4


# ── networks ────────────────────────────────────────────────────

class PlasticMLP(nn.Module):
    """MLP with optional spectral norm + layer norm for plasticity.

    Drop-in replacement for tianshou's Net — same interface.
    """

    def __init__(self, input_dim, hidden_sizes, device,
                 spectral_norm=False, layer_norm=False):
        super().__init__()
        self.output_dim = hidden_sizes[-1]
        self.device = device
        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            linear = nn.Linear(in_dim, h)
            if spectral_norm:
                linear = nn.utils.spectral_norm(linear)
            layers.append(linear)
            if layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            in_dim = h
        self.net = nn.Sequential(*layers).to(device)

    def forward(self, obs, state=None, info={}):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        return self.net(obs), state


def _make_net(obs_n, hidden, cfg):
    """Build backbone: PlasticMLP if spectral/layer norm, else tianshou Net."""
    if cfg.get("spectral_norm") or cfg.get("layer_norm"):
        return PlasticMLP(obs_n, hidden, DEVICE,
                          spectral_norm=cfg.get("spectral_norm", False),
                          layer_norm=cfg.get("layer_norm", False))
    return Net(state_shape=obs_n, hidden_sizes=hidden, device=DEVICE)


# ── wrappers for dict obs ────────────────────────────────────────

class MaskedSACActor(nn.Module):
    def __init__(self, net, act_n, device):
        super().__init__()
        self.net = net
        self.last = nn.Linear(net.output_dim, act_n, device=device)

    def forward(self, obs, state=None, info={}):
        mask = obs.mask if hasattr(obs, "mask") else None
        real_obs = obs.obs if hasattr(obs, "obs") else obs
        hidden, state = self.net(real_obs, state=state)
        logits = self.last(hidden)
        if mask is not None:
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
            logits = logits.masked_fill(~mask_t, -1e8)
        return logits, state


class ObsExtractCritic(nn.Module):
    def __init__(self, net, act_n, device):
        super().__init__()
        self.net = net
        self.last = nn.Linear(net.output_dim, act_n, device=device)

    def forward(self, obs, **kwargs):
        real_obs = obs.obs if hasattr(obs, "obs") else obs
        hidden, _ = self.net(real_obs)
        return self.last(hidden)


def load_config(path=None):
    """Load config from JSON, falling back to defaults for missing keys."""
    with open(DEFAULT_CONFIG) as f:
        cfg = json.load(f)
    if path is not None:
        with open(path) as f:
            overrides = json.load(f)
        cfg.update(overrides)
    return cfg


def run_trial(cfg: dict, trial_epochs: int) -> float:
    """Short training trial for hyperparameter search.

    No weights saved, no tensorboard. Returns best test_reward seen.
    """
    obs_n, act_n = 78, 40
    hidden = cfg["hidden"]

    net_a = _make_net(obs_n, hidden, cfg)
    net_c1 = _make_net(obs_n, hidden, cfg)
    net_c2 = _make_net(obs_n, hidden, cfg)

    actor = MaskedSACActor(net_a, act_n, DEVICE).to(DEVICE)
    critic1 = ObsExtractCritic(net_c1, act_n, DEVICE).to(DEVICE)
    critic2 = ObsExtractCritic(net_c2, act_n, DEVICE).to(DEVICE)

    actor_optim = torch.optim.Adam(actor.parameters(), lr=cfg["lr_actor"])
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=cfg["lr_critic"])
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=cfg["lr_critic"])

    target_entropy = -np.log(1.0 / act_n) * cfg["entropy_frac"]
    log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
    alpha_optim = torch.optim.Adam([log_alpha], lr=cfg["lr_actor"])
    alpha = (target_entropy, log_alpha, alpha_optim)

    policy = DiscreteSACPolicy(
        actor, actor_optim,
        critic1, critic1_optim,
        critic2, critic2_optim,
        tau=cfg["tau"],
        gamma=cfg["gamma"],
        alpha=alpha,
        estimation_step=int(cfg["n_step"]),
    )

    train_envs = DummyVectorEnv([lambda: TetrisEnv() for _ in range(NUM_TRAIN)])
    test_envs = DummyVectorEnv([lambda: TetrisEnv() for _ in range(NUM_TEST)])

    per_alpha = cfg.get("per_alpha", 0.0)
    if per_alpha > 0:
        buf = PrioritizedVectorReplayBuffer(
            int(cfg["buffer_size"]), NUM_TRAIN,
            alpha=per_alpha, beta=cfg.get("per_beta", 0.4),
        )
    else:
        buf = VectorReplayBuffer(int(cfg["buffer_size"]), NUM_TRAIN)
    train_collector = Collector(policy, train_envs, buf)
    test_collector = Collector(policy, test_envs)

    reset_freq = int(cfg.get("reset_freq", 0))
    _last_reset = [-1]

    def _reset_heads(epoch):
        if reset_freq and epoch % reset_freq == 0 and epoch != _last_reset[0]:
            _last_reset[0] = epoch
            actor.last.reset_parameters()

    result = offpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=trial_epochs,
        step_per_epoch=int(cfg["step_per_epoch"]),
        step_per_collect=int(cfg["step_per_collect"]),
        update_per_step=cfg["update_per_step"],
        episode_per_test=int(cfg["episode_per_test"]),
        batch_size=int(cfg["batch_size"]),
        train_fn=lambda epoch, env_step: _reset_heads(epoch),
        save_best_fn=lambda p: None,
        save_checkpoint_fn=lambda e, es, gs: None,
        verbose=False,
    )

    train_envs.close()
    test_envs.close()
    return float(result.get("best_reward", result.get("rew", 0.0)))


def main(max_epoch=None, config_path=None):
    cfg = load_config(config_path)
    if max_epoch is not None:
        cfg["max_epoch"] = max_epoch

    print(f"SAC v2 config: {json.dumps(cfg, indent=2)}")

    from datetime import datetime
    from torch.utils.tensorboard import SummaryWriter
    from tianshou.utils import TensorboardLogger
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = WEIGHTS_DIR.parent / "logs" / "sac_v2" / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    logger = TensorboardLogger(SummaryWriter(str(log_dir)))

    # ── environments ─────────────────────────────────────────────
    train_envs = DummyVectorEnv([lambda: TetrisEnv() for _ in range(NUM_TRAIN)])
    test_envs = DummyVectorEnv([lambda: TetrisEnv() for _ in range(NUM_TEST)])

    obs_n = 78
    act_n = 40
    hidden = cfg["hidden"]

    # ── networks ─────────────────────────────────────────────────
    net_a = _make_net(obs_n, hidden, cfg)
    net_c1 = _make_net(obs_n, hidden, cfg)
    net_c2 = _make_net(obs_n, hidden, cfg)

    actor = MaskedSACActor(net_a, act_n, DEVICE).to(DEVICE)
    critic1 = ObsExtractCritic(net_c1, act_n, DEVICE).to(DEVICE)
    critic2 = ObsExtractCritic(net_c2, act_n, DEVICE).to(DEVICE)

    actor_optim = torch.optim.Adam(actor.parameters(), lr=cfg["lr_actor"])
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=cfg["lr_critic"])
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=cfg["lr_critic"])

    # ── auto-tune alpha ──────────────────────────────────────────
    target_entropy = -np.log(1.0 / act_n) * cfg["entropy_frac"]
    log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
    alpha_optim = torch.optim.Adam([log_alpha], lr=cfg["lr_actor"])
    alpha = (target_entropy, log_alpha, alpha_optim)

    # ── policy ───────────────────────────────────────────────────
    policy = DiscreteSACPolicy(
        actor, actor_optim,
        critic1, critic1_optim,
        critic2, critic2_optim,
        tau=cfg["tau"],
        gamma=cfg["gamma"],
        alpha=alpha,
        estimation_step=int(cfg["n_step"]),
    )

    # ── collectors ───────────────────────────────────────────────
    per_alpha = cfg.get("per_alpha", 0.0)
    if per_alpha > 0:
        buf = PrioritizedVectorReplayBuffer(
            int(cfg["buffer_size"]), NUM_TRAIN,
            alpha=per_alpha, beta=cfg.get("per_beta", 0.4),
        )
    else:
        buf = VectorReplayBuffer(int(cfg["buffer_size"]), NUM_TRAIN)
    train_collector = Collector(policy, train_envs, buf)
    test_collector = Collector(policy, test_envs)

    # ── train ────────────────────────────────────────────────────
    reset_freq = int(cfg.get("reset_freq", 0))
    _last_reset_epoch = [-1]

    def train_fn(epoch, env_step):
        if reset_freq and epoch % reset_freq == 0 and epoch != _last_reset_epoch[0]:
            _last_reset_epoch[0] = epoch
            actor.last.reset_parameters()
            print(f"  -> reset actor head (epoch {epoch})")

    run_weights_dir = WEIGHTS_DIR / "sac_v2" / run_name
    run_weights_dir.mkdir(parents=True, exist_ok=True)

    def save_best(policy):
        path = run_weights_dir / "best.pth"
        torch.save(policy.state_dict(), path)
        with open(run_weights_dir / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"  -> saved weights/{run_name}/best.pth")

    checkpoint_freq = int(cfg["checkpoint_freq"])

    def save_checkpoint(epoch, env_step, gradient_step):
        if epoch % checkpoint_freq == 0:
            path = run_weights_dir / f"ep{epoch:04d}.pth"
            torch.save(policy.state_dict(), path)
            print(f"  -> checkpoint: {path.name}")

    result = offpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=int(cfg["max_epoch"]),
        step_per_epoch=int(cfg["step_per_epoch"]),
        step_per_collect=int(cfg["step_per_collect"]),
        update_per_step=cfg["update_per_step"],
        episode_per_test=int(cfg["episode_per_test"]),
        batch_size=int(cfg["batch_size"]),
        train_fn=train_fn,
        save_best_fn=save_best,
        save_checkpoint_fn=save_checkpoint,
        logger=logger,
        verbose=True,
    )

    print(f"\nTraining complete. Result: {result}")


if __name__ == "__main__":
    main()
