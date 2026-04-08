"""Train a Discrete SAC agent on Tetris using tianshou (0.5.x).

DiscreteSACPolicy doesn't natively extract obs.obs from dict obs,
so we wrap the actor and critics.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

WEIGHTS_DIR = Path(__file__).parent / "weights"
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import DiscreteSACPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic

from tetris.env import TetrisEnv

# ── hyper-parameters ─────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_TRAIN = 8
NUM_TEST = 4
HIDDEN = [256, 256]
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
GAMMA = 0.99
TAU = 0.005
N_STEP = 3
ALPHA = 0.2  # entropy coefficient (auto-tuned)
BATCH_SIZE = 256
BUFFER_SIZE = 100_000
STEP_PER_EPOCH = 10_000
STEP_PER_COLLECT = 100
UPDATE_PER_STEP = 0.3
MAX_EPOCH = 400
EPISODE_PER_TEST = 10
CHECKPOINT_FREQ = 50   # save a checkpoint every N epochs


# ── wrappers for dict obs ───────────────────────────────────────

class MaskedSACActor(nn.Module):
    """Extract obs.obs and apply action mask to logits."""

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
    """Extract obs.obs, output Q-values for all actions."""

    def __init__(self, net, act_n, device):
        super().__init__()
        self.net = net
        self.last = nn.Linear(net.output_dim, act_n, device=device)

    def forward(self, obs, **kwargs):
        real_obs = obs.obs if hasattr(obs, "obs") else obs
        hidden, _ = self.net(real_obs)
        return self.last(hidden)


def main(max_epoch=MAX_EPOCH):
    from torch.utils.tensorboard import SummaryWriter
    from tianshou.utils import TensorboardLogger
    log_dir = WEIGHTS_DIR.parent / "logs" / "sac"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = TensorboardLogger(SummaryWriter(str(log_dir)))

    # ── environments ─────────────────────────────────────────────
    train_envs = DummyVectorEnv([lambda: TetrisEnv() for _ in range(NUM_TRAIN)])
    test_envs = DummyVectorEnv([lambda: TetrisEnv() for _ in range(NUM_TEST)])

    obs_n = 78
    act_n = 40

    # ── networks ─────────────────────────────────────────────────
    net_a = Net(state_shape=obs_n, hidden_sizes=HIDDEN, device=DEVICE)
    net_c1 = Net(state_shape=obs_n, hidden_sizes=HIDDEN, device=DEVICE)
    net_c2 = Net(state_shape=obs_n, hidden_sizes=HIDDEN, device=DEVICE)

    actor = MaskedSACActor(net_a, act_n, DEVICE).to(DEVICE)
    critic1 = ObsExtractCritic(net_c1, act_n, DEVICE).to(DEVICE)
    critic2 = ObsExtractCritic(net_c2, act_n, DEVICE).to(DEVICE)

    actor_optim = torch.optim.Adam(actor.parameters(), lr=LR_ACTOR)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=LR_CRITIC)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=LR_CRITIC)

    # ── auto-tune alpha ──────────────────────────────────────────
    target_entropy = -np.log(1.0 / act_n) * 0.98  # ~98% of max entropy
    log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
    alpha_optim = torch.optim.Adam([log_alpha], lr=LR_ACTOR)
    alpha = (ALPHA, log_alpha, alpha_optim)

    # ── policy ───────────────────────────────────────────────────
    policy = DiscreteSACPolicy(
        actor, actor_optim,
        critic1, critic1_optim,
        critic2, critic2_optim,
        tau=TAU,
        gamma=GAMMA,
        alpha=alpha,
        estimation_step=N_STEP,
    )

    # ── collectors ───────────────────────────────────────────────
    train_collector = Collector(
        policy, train_envs,
        VectorReplayBuffer(BUFFER_SIZE, NUM_TRAIN),
    )
    test_collector = Collector(policy, test_envs)

    # ── train ────────────────────────────────────────────────────
    def save_best(policy):
        WEIGHTS_DIR.mkdir(exist_ok=True)
        torch.save(policy.state_dict(), WEIGHTS_DIR / "best_sac.pth")
        print(f"  -> saved rl_training/weights/best_sac.pth")

    ckpt_dir = WEIGHTS_DIR / "checkpoints"

    def save_checkpoint(epoch, env_step, gradient_step):
        if epoch % CHECKPOINT_FREQ == 0:
            ckpt_dir.mkdir(exist_ok=True)
            path = ckpt_dir / f"sac_ep{epoch:04d}.pth"
            torch.save(policy.state_dict(), path)
            print(f"  -> checkpoint: {path.name}")

    result = offpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=max_epoch,
        step_per_epoch=STEP_PER_EPOCH,
        step_per_collect=STEP_PER_COLLECT,
        update_per_step=UPDATE_PER_STEP,
        episode_per_test=EPISODE_PER_TEST,
        batch_size=BATCH_SIZE,
        save_best_fn=save_best,
        save_checkpoint_fn=save_checkpoint,
        logger=logger,
        verbose=True,
    )

    print(f"\nTraining complete. Result: {result}")


if __name__ == "__main__":
    main()
