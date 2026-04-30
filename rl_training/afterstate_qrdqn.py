"""Afterstate quantile-regression DQN — Stage 4 Branch B.

Differs from afterstate_dqn.py in three places:
- Network outputs N quantiles per state instead of a scalar V.
- Greedy action selection picks argmax over quantile means (E[V]).
- Loss is the Dabney et al. quantile Huber over per-state distributions.

Tests whether distributional V helps when the underlying afterstate paradigm
already works. Same 4-feature input, same nuno reward, same target-net update.

Usage:
    python train.py --algo afterstate_qrdqn
    python train.py --algo afterstate_qrdqn --config rl_training/configs/afterstate_qrdqn.json
"""

import json
import random
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from tetris.env.afterstate import (
    apply_action,
    enumerate_afterstates,
    nuno_reward,
    state_props,
)
from tetris.game import Tetris

WEIGHTS_DIR = Path(__file__).parent / "weights"
LOGS_DIR = Path(__file__).parent / "logs"
DEFAULT_CONFIG = Path(__file__).parent / "configs" / "afterstate_qrdqn.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── quantile value network ────────────────────────────────────────

class QuantileVNet(nn.Module):
    """MLP mapping 4-feature afterstate -> N quantiles of V."""

    def __init__(self, hidden, n_quantiles, device):
        super().__init__()
        self.n_quantiles = n_quantiles
        layers, in_dim = [], 4
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_quantiles))
        self.net = nn.Sequential(*layers).to(device)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32,
                                device=next(self.parameters()).device)
        return self.net(x)  # [batch, n_quantiles]

    def value(self, x):
        return self.forward(x).mean(dim=-1)


# ── replay buffer ─────────────────────────────────────────────────

class Replay:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, s, s_next, r, done):
        self.buf.append((s, s_next, r, done))

    def sample(self, batch_size):
        idx = np.random.randint(0, len(self.buf), size=batch_size)
        batch = [self.buf[i] for i in idx]
        s = np.stack([b[0] for b in batch])
        s_next = np.stack([b[1] for b in batch])
        r = np.array([b[2] for b in batch], dtype=np.float32)
        done = np.array([b[3] for b in batch], dtype=np.float32)
        return s, s_next, r, done

    def __len__(self):
        return len(self.buf)


# ── action selection ──────────────────────────────────────────────

def choose_action(game, net, eps):
    afterstates = enumerate_afterstates(game)
    if not afterstates:
        return None, None
    if random.random() < eps:
        action = random.choice(list(afterstates.keys()))
        return action, afterstates[action][0]
    actions = list(afterstates.keys())
    feats = np.stack([afterstates[a][0] for a in actions])
    with torch.no_grad():
        values = net.value(feats).cpu().numpy()
    best_idx = int(np.argmax(values))
    return actions[best_idx], afterstates[actions[best_idx]][0]


# ── episode runner ────────────────────────────────────────────────

def run_episode(net, eps, replay, gamma, max_steps=10000):
    game = Tetris()
    game.next_block()
    current_state = state_props(game.field, 0)
    total_reward = 0.0
    total_lines = 0
    steps = 0
    while game.state != "gameover" and steps < max_steps:
        action, chosen_features = choose_action(game, net, eps)
        if action is None:
            break
        lines, done = apply_action(game, action)
        reward = nuno_reward(lines, done)
        if replay is not None:
            replay.push(current_state, chosen_features, reward, done)
        current_state = chosen_features
        total_reward += reward
        total_lines += lines
        steps += 1
        if done:
            break
    return total_reward, steps, total_lines


def evaluate(net, n_episodes=10):
    rewards, lines_list, steps_list = [], [], []
    for _ in range(n_episodes):
        r, s, l = run_episode(net, eps=0.0, replay=None, gamma=0.0)
        rewards.append(r)
        steps_list.append(s)
        lines_list.append(l)
    return float(np.mean(rewards)), float(np.mean(lines_list)), float(np.mean(steps_list))


# ── quantile huber loss ───────────────────────────────────────────

def quantile_huber_loss(pred, target, kappa=1.0):
    """Dabney et al. QR-DQN loss. pred/target shape [B, N]; sum over predicted
    quantile dim, mean over target sample dim, mean over batch."""
    n = pred.shape[1]
    device = pred.device
    tau = (torch.arange(n, device=device, dtype=torch.float32) + 0.5) / n
    tau = tau.view(1, n, 1)

    diff = target.unsqueeze(1) - pred.unsqueeze(2)  # [B, N_pred, N_target]
    abs_diff = diff.abs()
    huber = torch.where(
        abs_diff <= kappa,
        0.5 * diff.pow(2),
        kappa * (abs_diff - 0.5 * kappa),
    )
    weight = (tau - (diff.detach() < 0).float()).abs()
    return (weight * huber).sum(dim=1).mean(dim=1).mean()


# ── training ──────────────────────────────────────────────────────

def train_step(net, target_net, optim, replay, batch_size, gamma, kappa):
    s, s_next, r, done = replay.sample(batch_size)
    device = next(net.parameters()).device
    s_t = torch.as_tensor(s, dtype=torch.float32, device=device)
    s_next_t = torch.as_tensor(s_next, dtype=torch.float32, device=device)
    r_t = torch.as_tensor(r, dtype=torch.float32, device=device).unsqueeze(-1)
    done_t = torch.as_tensor(done, dtype=torch.float32, device=device).unsqueeze(-1)

    with torch.no_grad():
        z_next = target_net(s_next_t)                       # [B, N]
        target = r_t + gamma * z_next * (1.0 - done_t)      # [B, N]
    pred = net(s_t)                                         # [B, N]
    loss = quantile_huber_loss(pred, target, kappa)

    optim.zero_grad()
    loss.backward()
    optim.step()
    return float(loss.item())


def load_config(path=None):
    with open(DEFAULT_CONFIG) as f:
        cfg = json.load(f)
    if path is not None:
        with open(path) as f:
            cfg.update(json.load(f))
    return cfg


def run_trial(cfg, trial_episodes):
    net = QuantileVNet(cfg["hidden"], cfg["n_quantiles"], DEVICE)
    target_net = QuantileVNet(cfg["hidden"], cfg["n_quantiles"], DEVICE)
    target_net.load_state_dict(net.state_dict())
    optim = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    replay = Replay(int(cfg["buffer_size"]))

    eps = cfg["eps_start"]
    eps_end = cfg["eps_end"]
    decay = (eps - eps_end) / max(cfg["eps_decay_episodes"], 1)
    best = -float("inf")

    for ep in range(trial_episodes):
        run_episode(net, eps, replay, cfg["gamma"])
        if len(replay) >= cfg["batch_size"]:
            for _ in range(cfg["updates_per_episode"]):
                train_step(net, target_net, optim, replay,
                           cfg["batch_size"], cfg["gamma"], cfg["kappa"])
        if ep % cfg["target_update_freq"] == 0:
            target_net.load_state_dict(net.state_dict())
        if eps > eps_end:
            eps = max(eps_end, eps - decay)
        if (ep + 1) % cfg["eval_freq"] == 0:
            mean_r, _, _ = evaluate(net, cfg["eval_episodes"])
            best = max(best, mean_r)
    return best


def main(max_epoch=None, config_path=None):
    cfg = load_config(config_path)
    if max_epoch is not None:
        cfg["episodes"] = max_epoch
    print(f"Afterstate QR-DQN config: {json.dumps(cfg, indent=2)}")

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = LOGS_DIR / "afterstate_qrdqn" / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = WEIGHTS_DIR / "afterstate_qrdqn" / run_name
    weights_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    with open(weights_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    writer = SummaryWriter(str(log_dir))

    net = QuantileVNet(cfg["hidden"], cfg["n_quantiles"], DEVICE)
    target_net = QuantileVNet(cfg["hidden"], cfg["n_quantiles"], DEVICE)
    target_net.load_state_dict(net.state_dict())
    optim = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    replay = Replay(int(cfg["buffer_size"]))

    eps = cfg["eps_start"]
    eps_end = cfg["eps_end"]
    decay = (eps - eps_end) / max(cfg["eps_decay_episodes"], 1)
    best_eval = -float("inf")

    for ep in range(int(cfg["episodes"])):
        ep_reward, ep_steps, ep_lines = run_episode(net, eps, replay, cfg["gamma"])
        writer.add_scalar("train/reward", ep_reward, ep)
        writer.add_scalar("train/lines", ep_lines, ep)
        writer.add_scalar("train/steps", ep_steps, ep)
        writer.add_scalar("train/eps", eps, ep)

        if len(replay) >= cfg["batch_size"]:
            losses = []
            for _ in range(cfg["updates_per_episode"]):
                losses.append(train_step(net, target_net, optim, replay,
                                         cfg["batch_size"], cfg["gamma"], cfg["kappa"]))
            writer.add_scalar("train/loss", float(np.mean(losses)), ep)

        if ep % cfg["target_update_freq"] == 0:
            target_net.load_state_dict(net.state_dict())

        if eps > eps_end:
            eps = max(eps_end, eps - decay)

        if (ep + 1) % cfg["eval_freq"] == 0:
            mean_r, mean_lines, mean_steps = evaluate(net, cfg["eval_episodes"])
            writer.add_scalar("eval/reward", mean_r, ep)
            writer.add_scalar("eval/lines", mean_lines, ep)
            writer.add_scalar("eval/steps", mean_steps, ep)
            print(f"ep {ep+1}/{cfg['episodes']} | eval reward {mean_r:.2f} | "
                  f"lines {mean_lines:.1f} | steps {mean_steps:.0f} | eps {eps:.3f}")
            if mean_r > best_eval:
                best_eval = mean_r
                torch.save(net.state_dict(), weights_dir / "best.pth")

        if (ep + 1) % cfg["checkpoint_freq"] == 0:
            torch.save(net.state_dict(), weights_dir / f"ep{ep+1:04d}.pth")

    writer.close()
    print(f"\nDone. Best eval reward: {best_eval:.2f}")
    return best_eval


if __name__ == "__main__":
    main()
