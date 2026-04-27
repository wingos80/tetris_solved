"""Afterstate value-based DQN (nuno-faria/tetris-ai reproduction).

Differs from dqn.py:
- Network outputs single scalar V(afterstate), not Q(s,a) per action.
- Action selection enumerates all valid placements, picks max V.
- State = 4-feature hand-crafted vector [lines_cleared, holes, bumpiness, total_height].
- Reward = 1 + lines^2 * width - 2 if gameover (per placement).
- Standalone training loop — tianshou's offpolicy_trainer assumes Q(s,a).

Usage:
    python train.py --algo afterstate
    python train.py --algo afterstate --config rl_training/configs/afterstate_dqn.json
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
DEFAULT_CONFIG = Path(__file__).parent / "configs" / "afterstate_dqn.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── value network ─────────────────────────────────────────────────

class VNet(nn.Module):
    """MLP mapping 4-feature afterstate -> scalar value."""

    def __init__(self, hidden, device):
        super().__init__()
        layers, in_dim = [], 4
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers).to(device)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32,
                                device=next(self.parameters()).device)
        return self.net(x).squeeze(-1)


# ── replay buffer ─────────────────────────────────────────────────

class Replay:
    """Stores (s, s', r, done). s and s' are 4-d feature vectors."""

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
    """Epsilon-greedy over enumerated afterstates. Returns (action, chosen_features)."""
    afterstates = enumerate_afterstates(game)
    if not afterstates:
        return None, None
    if random.random() < eps:
        action = random.choice(list(afterstates.keys()))
        return action, afterstates[action][0]
    # Greedy: batch all candidate features through net
    actions = list(afterstates.keys())
    feats = np.stack([afterstates[a][0] for a in actions])
    with torch.no_grad():
        values = net(feats).cpu().numpy()
    best_idx = int(np.argmax(values))
    return actions[best_idx], afterstates[actions[best_idx]][0]


# ── episode runner ────────────────────────────────────────────────

def run_episode(net, eps, replay, gamma, max_steps=10000):
    """Play one episode, push transitions to replay. Returns (total_reward, steps, lines)."""
    game = Tetris()
    game.next_block()
    # Initial afterstate = empty board (zeros)
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
    """Greedy rollouts. Returns mean reward and mean lines."""
    rewards, lines_list, steps_list = [], [], []
    for _ in range(n_episodes):
        r, s, l = run_episode(net, eps=0.0, replay=None, gamma=0.0)
        rewards.append(r)
        steps_list.append(s)
        lines_list.append(l)
    return float(np.mean(rewards)), float(np.mean(lines_list)), float(np.mean(steps_list))


# ── training ──────────────────────────────────────────────────────

def train_step(net, target_net, optim, replay, batch_size, gamma):
    s, s_next, r, done = replay.sample(batch_size)
    device = next(net.parameters()).device
    s_t = torch.as_tensor(s, dtype=torch.float32, device=device)
    s_next_t = torch.as_tensor(s_next, dtype=torch.float32, device=device)
    r_t = torch.as_tensor(r, dtype=torch.float32, device=device)
    done_t = torch.as_tensor(done, dtype=torch.float32, device=device)

    with torch.no_grad():
        v_next = target_net(s_next_t)
        target = r_t + gamma * v_next * (1.0 - done_t)
    pred = net(s_t)
    loss = nn.functional.mse_loss(pred, target)

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
    """Short training trial — no logging, no checkpoints. Returns best mean eval reward."""
    net = VNet(cfg["hidden"], DEVICE)
    target_net = VNet(cfg["hidden"], DEVICE)
    target_net.load_state_dict(net.state_dict())
    optim = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    replay = Replay(int(cfg["buffer_size"]))

    eps = cfg["eps_start"]
    eps_end = cfg["eps_end"]
    eps_decay_episodes = cfg["eps_decay_episodes"]
    decay = (eps - eps_end) / max(eps_decay_episodes, 1)
    best = -float("inf")

    for ep in range(trial_episodes):
        run_episode(net, eps, replay, cfg["gamma"])
        if len(replay) >= cfg["batch_size"]:
            for _ in range(cfg["updates_per_episode"]):
                train_step(net, target_net, optim,
                           replay, cfg["batch_size"], cfg["gamma"])
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
    print(f"Afterstate DQN config: {json.dumps(cfg, indent=2)}")

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = LOGS_DIR / "afterstate" / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = WEIGHTS_DIR / "afterstate" / run_name
    weights_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    with open(weights_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    writer = SummaryWriter(str(log_dir))

    net = VNet(cfg["hidden"], DEVICE)
    target_net = VNet(cfg["hidden"], DEVICE)
    target_net.load_state_dict(net.state_dict())
    optim = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    replay = Replay(int(cfg["buffer_size"]))

    eps = cfg["eps_start"]
    eps_end = cfg["eps_end"]
    eps_decay_episodes = cfg["eps_decay_episodes"]
    decay = (eps - eps_end) / max(eps_decay_episodes, 1)
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
                losses.append(train_step(net, target_net, optim,
                                         replay, cfg["batch_size"], cfg["gamma"]))
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
