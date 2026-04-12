"""Train a masked-PPO agent on Tetris using tianshou (0.5.x)."""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

WEIGHTS_DIR = Path(__file__).parent / "weights"
from torch.distributions import Categorical
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic

from tetris.env import TetrisEnv

# ── hyper-parameters ─────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_TRAIN = 8
NUM_TEST = 4
HIDDEN = [256, 256]
LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
EPS_CLIP = 0.2
VF_COEF = 0.5
ENT_COEF = 0.01
MAX_GRAD_NORM = 0.5
BATCH_SIZE = 256
BUFFER_SIZE = 20_000
STEP_PER_EPOCH = 10_000
STEP_PER_COLLECT = 2_000
REPEAT_PER_COLLECT = 4
MAX_EPOCH = 200
EPISODE_PER_TEST = 10
CHECKPOINT_FREQ = 25   # save a checkpoint every N epochs


# ── masked wrappers ─────────────────────────────────────────────

class MaskedActor(nn.Module):
    """Wraps a tianshou Actor to apply action masks before distribution."""

    def __init__(self, actor):
        super().__init__()
        self.actor = actor  # Actor(softmax_output=False) → raw logits

    def forward(self, obs, state=None, info={}):
        mask = obs.mask if hasattr(obs, "mask") else None
        real_obs = obs.obs if hasattr(obs, "obs") else obs
        logits, hidden = self.actor(real_obs, state, info)
        if mask is not None:
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
            logits = logits.masked_fill(~mask_t, -1e8)
        return logits, hidden


class ObsExtractCritic(nn.Module):
    """Wraps a tianshou Critic to extract 'obs' from dict observations."""

    def __init__(self, critic):
        super().__init__()
        self.critic = critic

    def forward(self, obs, **kwargs):
        real_obs = obs.obs if hasattr(obs, "obs") else obs
        return self.critic(real_obs, **kwargs)


def main(max_epoch=MAX_EPOCH):
    from datetime import datetime
    from torch.utils.tensorboard import SummaryWriter
    from tianshou.utils import TensorboardLogger
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = WEIGHTS_DIR.parent / "logs" / "ppo" / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = TensorboardLogger(SummaryWriter(str(log_dir)))

    # ── environments ─────────────────────────────────────────────
    train_envs = DummyVectorEnv([lambda: TetrisEnv() for _ in range(NUM_TRAIN)])
    test_envs = DummyVectorEnv([lambda: TetrisEnv() for _ in range(NUM_TEST)])

    obs_n = 78   # feature vector size (obs part of dict)
    act_n = 40   # 4 rotations * 10 columns

    # ── networks (separate backbones for actor & critic) ─────────
    net_a = Net(state_shape=obs_n, hidden_sizes=HIDDEN, device=DEVICE)
    net_c = Net(state_shape=obs_n, hidden_sizes=HIDDEN, device=DEVICE)
    raw_actor = Actor(net_a, act_n, device=DEVICE, softmax_output=False).to(DEVICE)
    raw_critic = Critic(net_c, device=DEVICE).to(DEVICE)

    actor = MaskedActor(raw_actor).to(DEVICE)
    critic = ObsExtractCritic(raw_critic).to(DEVICE)

    optim = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=LR,
    )

    # ── policy ───────────────────────────────────────────────────
    policy = PPOPolicy(
        actor, critic, optim,
        dist_fn=lambda logits: Categorical(logits=logits),
        discount_factor=GAMMA,
        gae_lambda=GAE_LAMBDA,
        eps_clip=EPS_CLIP,
        vf_coef=VF_COEF,
        ent_coef=ENT_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        action_space=TetrisEnv().action_space,
    )

    # ── collectors ───────────────────────────────────────────────
    train_collector = Collector(
        policy, train_envs,
        VectorReplayBuffer(BUFFER_SIZE, NUM_TRAIN),
    )
    test_collector = Collector(policy, test_envs)

    # ── train ────────────────────────────────────────────────────
    run_weights_dir = WEIGHTS_DIR / "ppo" / run_name
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

    result = onpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=max_epoch,
        step_per_epoch=STEP_PER_EPOCH,
        step_per_collect=STEP_PER_COLLECT,
        repeat_per_collect=REPEAT_PER_COLLECT,
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
