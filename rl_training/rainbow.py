"""Train a Rainbow DQN agent on Tetris using tianshou (0.5.x).

Rainbow = C51 (distributional) + Double DQN + PER + Dueling + N-step + NoisyNet.
Native action masking via dict obs (inherited from DQNPolicy).
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

WEIGHTS_DIR = Path(__file__).parent / "weights"
from tianshou.data import Collector, PrioritizedVectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import RainbowPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.discrete import NoisyLinear

from tetris.env import TetrisEnv

# ── hyper-parameters ─────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_TRAIN = 8
NUM_TEST = 4
HIDDEN = 256
LR = 6.25e-5
GAMMA = 0.99
N_STEP = 3
NUM_ATOMS = 51
V_MIN = -20.0
V_MAX = 25.0
TARGET_UPDATE_FREQ = 500
BATCH_SIZE = 256
BUFFER_SIZE = 100_000
PER_ALPHA = 0.6
PER_BETA = 0.4
STEP_PER_EPOCH = 10_000
STEP_PER_COLLECT = 100
UPDATE_PER_STEP = 0.3
MAX_EPOCH = 400
EPISODE_PER_TEST = 10
CHECKPOINT_FREQ = 50   # save a checkpoint every N epochs


# ── Rainbow network (dueling + noisy + distributional) ───────────

class RainbowNet(nn.Module):
    """Dueling network with NoisyLinear layers and C51 atom output.

    obs → shared_fc → [value_stream, advantage_stream] → atoms
    Each stream uses NoisyLinear for exploration.
    """

    def __init__(self, obs_n, act_n, hidden, num_atoms, device):
        super().__init__()
        self.act_n = act_n
        self.num_atoms = num_atoms

        # Shared feature extraction (regular linear — noise in heads only)
        self.feature = nn.Sequential(
            nn.Linear(obs_n, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        ).to(device)

        # Value stream (NoisyLinear)
        self.value = nn.Sequential(
            NoisyLinear(hidden, hidden),
            nn.ReLU(),
            NoisyLinear(hidden, num_atoms),
        ).to(device)

        # Advantage stream (NoisyLinear)
        self.advantage = nn.Sequential(
            NoisyLinear(hidden, hidden),
            nn.ReLU(),
            NoisyLinear(hidden, act_n * num_atoms),
        ).to(device)

    def forward(self, obs, state=None, info={}):
        # Handle dict obs (tianshou extracts obs.obs in DQNPolicy.forward,
        # but during learn() the model may receive raw tensors)
        if hasattr(obs, "obs"):
            obs = obs.obs
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=next(self.parameters()).device)

        feat = self.feature(obs)
        value = self.value(feat).view(-1, 1, self.num_atoms)         # (B, 1, atoms)
        adv = self.advantage(feat).view(-1, self.act_n, self.num_atoms)  # (B, A, atoms)

        # Dueling: Q = V + A - mean(A)
        logits = value + adv - adv.mean(dim=1, keepdim=True)  # (B, A, atoms)
        # Softmax over atoms for each action → probability distribution
        probs = logits.softmax(dim=2)
        return probs, state


def main(max_epoch=MAX_EPOCH):
    from datetime import datetime
    from torch.utils.tensorboard import SummaryWriter
    from tianshou.utils import TensorboardLogger
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = WEIGHTS_DIR.parent / "logs" / "rainbow" / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = TensorboardLogger(SummaryWriter(str(log_dir)))

    # ── environments ─────────────────────────────────────────────
    train_envs = DummyVectorEnv([lambda: TetrisEnv() for _ in range(NUM_TRAIN)])
    test_envs = DummyVectorEnv([lambda: TetrisEnv() for _ in range(NUM_TEST)])

    obs_n = 78
    act_n = 40

    # ── network ──────────────────────────────────────────────────
    net = RainbowNet(obs_n, act_n, HIDDEN, NUM_ATOMS, DEVICE)
    optim = torch.optim.Adam(net.parameters(), lr=LR)

    # ── policy ───────────────────────────────────────────────────
    policy = RainbowPolicy(
        net, optim,
        discount_factor=GAMMA,
        num_atoms=NUM_ATOMS,
        v_min=V_MIN,
        v_max=V_MAX,
        estimation_step=N_STEP,
        target_update_freq=TARGET_UPDATE_FREQ,
    ).to(DEVICE)

    # ── collectors (PER buffer for Rainbow) ──────────────────────
    train_collector = Collector(
        policy, train_envs,
        PrioritizedVectorReplayBuffer(
            BUFFER_SIZE, NUM_TRAIN,
            alpha=PER_ALPHA, beta=PER_BETA,
        ),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs, exploration_noise=False)

    # ── train ────────────────────────────────────────────────────
    run_weights_dir = WEIGHTS_DIR / "rainbow" / run_name
    run_weights_dir.mkdir(parents=True, exist_ok=True)

    def save_best(policy):
        path = run_weights_dir / "best.pth"
        torch.save(policy.state_dict(), path)
        print(f"  -> saved weights/{run_name}/best.pth")

    def save_checkpoint(epoch, env_step, gradient_step):
        if epoch % CHECKPOINT_FREQ == 0:
            path = run_weights_dir / f"ep{epoch:04d}.pth"
            torch.save(policy.state_dict(), path)
            print(f"  -> checkpoint: {path.name}")

    # Rainbow uses NoisyNet for exploration — no epsilon schedule needed
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
