"""Afterstate value learning with SPR auxiliary loss — Stage 4 Branch D.

Extends Branch A (afterstate_cnn.py) with a self-predictive representation
(SPR) auxiliary loss. The encoder must predict the next board's latent
representation given the current latent + action embedding. This auxiliary
objective forces the CNN to learn dynamics-aware features, addressing the
sample-efficiency bottleneck observed in Branch A.

Architecture:
  Encoder:    20x10 board -> CNN -> z  (flat latent, dim latent_dim)
  V-head:     z -> scalar V
  Transition: z + action_emb(rot, dx) -> z_next_pred
  SPR loss:   cosine_sim(z_next_pred, sg(target_encoder(s_next)))

Action embedding: (rot, dx) pair where dx = x - (-3), both embedded as
  learned 8-d vectors and summed -> 16-d action context -> MLP -> latent_dim.

Loss = TD(MSE) + spr_coef * SPR(cosine)

Usage:
    python train.py --algo afterstate_spr
    python train.py --algo afterstate_spr --config rl_training/configs/afterstate_spr.json
"""

import json
import random
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from tetris.env.afterstate import (
    apply_action,
    enumerate_afterstates_raw,
    nuno_reward,
)
from tetris.game import Tetris

WEIGHTS_DIR = Path(__file__).parent / "weights"
LOGS_DIR = Path(__file__).parent / "logs"
DEFAULT_CONFIG = Path(__file__).parent / "configs" / "afterstate_spr.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BOARD_SHAPE = (Tetris.ROWS, Tetris.COLS)  # (20, 10)
MAX_ROT = 4
MAX_DX = Tetris.COLS + 3  # x in [-3, COLS-1] -> dx in [0, COLS+2]


# ── network ───────────────────────────────────────────────────────

class SPRNet(nn.Module):
    """CNN encoder + V-head + transition model for SPR auxiliary loss."""

    def __init__(self, conv_channels, fc_hidden, latent_dim, action_emb_dim, device):
        super().__init__()
        R, C = BOARD_SHAPE

        # ── encoder ──────────────────────────────────────────────
        conv_layers, in_ch = [], 1
        for ch in conv_channels:
            conv_layers += [nn.Conv2d(in_ch, ch, kernel_size=3, padding=1), nn.ReLU()]
            in_ch = ch
        self.encoder_conv = nn.Sequential(*conv_layers)
        flat_dim = (conv_channels[-1] if conv_channels else 1) * R * C
        enc_fc = []
        in_dim = flat_dim
        for h in fc_hidden:
            enc_fc += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        enc_fc.append(nn.Linear(in_dim, latent_dim))
        self.encoder_fc = nn.Sequential(*enc_fc)

        # ── V-head ────────────────────────────────────────────────
        self.v_head = nn.Linear(latent_dim, 1)

        # ── action embedding ──────────────────────────────────────
        self.rot_emb = nn.Embedding(MAX_ROT, action_emb_dim)
        self.dx_emb = nn.Embedding(MAX_DX, action_emb_dim)
        act_dim = action_emb_dim * 2

        # ── transition model: (z, act) -> z_next_pred ─────────────
        self.transition = nn.Sequential(
            nn.Linear(latent_dim + act_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        self.to(device)

    def encode(self, x):
        """Board tensor -> latent z. Handles [R,C], [B,R,C] inputs."""
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32,
                                device=next(self.parameters()).device)
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)   # [1,1,R,C]
        elif x.dim() == 3:
            x = x.unsqueeze(1)                # [B,1,R,C]
        feat = self.encoder_conv(x).flatten(1)
        return self.encoder_fc(feat)           # [B, latent_dim]

    def forward(self, x):
        """Returns scalar V for board input x."""
        z = self.encode(x)
        return self.v_head(z).squeeze(-1)      # [B]

    def predict_next_latent(self, z, rot, dx):
        """Predict next latent from current latent + action (rot, dx) indices."""
        device = next(self.parameters()).device
        if not isinstance(rot, torch.Tensor):
            rot = torch.as_tensor(rot, dtype=torch.long, device=device)
        if not isinstance(dx, torch.Tensor):
            dx = torch.as_tensor(dx, dtype=torch.long, device=device)
        act = torch.cat([self.rot_emb(rot), self.dx_emb(dx)], dim=-1)  # [B, 2*emb]
        return self.transition(torch.cat([z, act], dim=-1))              # [B, latent]


# ── replay buffer ─────────────────────────────────────────────────

class Replay:
    """Stores (s, rot, dx, s_next, r, done). s and s_next are [R,C] float32."""

    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, s, rot, dx, s_next, r, done):
        self.buf.append((s, rot, dx, s_next, r, done))

    def sample(self, batch_size):
        idx = np.random.randint(0, len(self.buf), size=batch_size)
        batch = [self.buf[i] for i in idx]
        s      = np.stack([b[0] for b in batch])
        rot    = np.array([b[1] for b in batch], dtype=np.int64)
        dx     = np.array([b[2] for b in batch], dtype=np.int64)
        s_next = np.stack([b[3] for b in batch])
        r      = np.array([b[4] for b in batch], dtype=np.float32)
        done   = np.array([b[5] for b in batch], dtype=np.float32)
        return s, rot, dx, s_next, r, done

    def __len__(self):
        return len(self.buf)


# ── action selection ──────────────────────────────────────────────

def choose_action(game, net, eps):
    afterstates = enumerate_afterstates_raw(game)
    if not afterstates:
        return None, None, None, None
    if random.random() < eps:
        action = random.choice(list(afterstates.keys()))
    else:
        actions = list(afterstates.keys())
        boards = np.stack([afterstates[a][0] for a in actions])
        with torch.no_grad():
            values = net(boards).cpu().numpy()
        action = actions[int(np.argmax(values))]
    board, _ = afterstates[action]
    rot, x = action
    dx = x + 3  # shift to [0, MAX_DX-1]
    return action, board, rot, dx


# ── episode runner ────────────────────────────────────────────────

def run_episode(net, eps, replay, max_steps=10000):
    game = Tetris()
    game.next_block()
    current_board = np.zeros(BOARD_SHAPE, dtype=np.float32)
    total_reward = 0.0
    total_lines = 0
    steps = 0
    while game.state != "gameover" and steps < max_steps:
        action, chosen_board, rot, dx = choose_action(game, net, eps)
        if action is None:
            break
        lines, done = apply_action(game, action)
        reward = nuno_reward(lines, done)
        if replay is not None:
            replay.push(current_board, rot, dx, chosen_board, reward, done)
        current_board = chosen_board
        total_reward += reward
        total_lines += lines
        steps += 1
        if done:
            break
    return total_reward, steps, total_lines


def evaluate(net, n_episodes=10):
    rewards, lines_list, steps_list = [], [], []
    for _ in range(n_episodes):
        r, s, l = run_episode(net, eps=0.0, replay=None)
        rewards.append(r)
        steps_list.append(s)
        lines_list.append(l)
    return float(np.mean(rewards)), float(np.mean(lines_list)), float(np.mean(steps_list))


# ── training ──────────────────────────────────────────────────────

def train_step(net, target_net, optim, replay, batch_size, gamma, spr_coef):
    s, rot, dx, s_next, r, done = replay.sample(batch_size)
    device = next(net.parameters()).device

    s_t      = torch.as_tensor(s,    dtype=torch.float32, device=device)
    s_next_t = torch.as_tensor(s_next, dtype=torch.float32, device=device)
    r_t      = torch.as_tensor(r,    dtype=torch.float32, device=device)
    done_t   = torch.as_tensor(done, dtype=torch.float32, device=device)
    rot_t    = torch.as_tensor(rot,  dtype=torch.long,    device=device)
    dx_t     = torch.as_tensor(dx,   dtype=torch.long,    device=device)

    # ── TD loss (MSE) ─────────────────────────────────────────────
    with torch.no_grad():
        v_next = target_net(s_next_t)
        td_target = r_t + gamma * v_next * (1.0 - done_t)
    v_pred = net(s_t)
    td_loss = F.mse_loss(v_pred, td_target)

    # ── SPR auxiliary loss ────────────────────────────────────────
    z_cur  = net.encode(s_t)                              # [B, latent]
    z_pred = net.predict_next_latent(z_cur, rot_t, dx_t)  # [B, latent]
    with torch.no_grad():
        z_next = target_net.encode(s_next_t)              # [B, latent] stop-grad
    # Cosine similarity loss: 1 - cos_sim (minimise -> maximise similarity)
    spr_loss = 1.0 - F.cosine_similarity(z_pred, z_next, dim=-1).mean()

    loss = td_loss + spr_coef * spr_loss

    optim.zero_grad()
    loss.backward()
    optim.step()
    return float(td_loss.item()), float(spr_loss.item())


def load_config(path=None):
    with open(DEFAULT_CONFIG) as f:
        cfg = json.load(f)
    if path is not None:
        with open(path) as f:
            cfg.update(json.load(f))
    return cfg


def _build_net(cfg, device):
    return SPRNet(
        cfg["conv_channels"],
        cfg["fc_hidden"],
        cfg["latent_dim"],
        cfg["action_emb_dim"],
        device,
    )


def run_trial(cfg, trial_episodes):
    net = _build_net(cfg, DEVICE)
    target_net = _build_net(cfg, DEVICE)
    target_net.load_state_dict(net.state_dict())
    optim = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    replay = Replay(int(cfg["buffer_size"]))

    eps = cfg["eps_start"]
    eps_end = cfg["eps_end"]
    decay = (eps - eps_end) / max(cfg["eps_decay_episodes"], 1)
    best = -float("inf")

    for ep in range(trial_episodes):
        run_episode(net, eps, replay)
        if len(replay) >= cfg["batch_size"]:
            for _ in range(cfg["updates_per_episode"]):
                train_step(net, target_net, optim, replay,
                           cfg["batch_size"], cfg["gamma"], cfg["spr_coef"])
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
    print(f"Afterstate SPR config: {json.dumps(cfg, indent=2)}")

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = LOGS_DIR / "afterstate_spr" / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = WEIGHTS_DIR / "afterstate_spr" / run_name
    weights_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    with open(weights_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    writer = SummaryWriter(str(log_dir))

    net = _build_net(cfg, DEVICE)
    target_net = _build_net(cfg, DEVICE)
    target_net.load_state_dict(net.state_dict())
    optim = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    replay = Replay(int(cfg["buffer_size"]))

    eps = cfg["eps_start"]
    eps_end = cfg["eps_end"]
    decay = (eps - eps_end) / max(cfg["eps_decay_episodes"], 1)
    best_eval = -float("inf")

    for ep in range(int(cfg["episodes"])):
        ep_reward, ep_steps, ep_lines = run_episode(net, eps, replay)
        writer.add_scalar("train/reward", ep_reward, ep)
        writer.add_scalar("train/lines", ep_lines, ep)
        writer.add_scalar("train/steps", ep_steps, ep)
        writer.add_scalar("train/eps", eps, ep)

        if len(replay) >= cfg["batch_size"]:
            td_losses, spr_losses = [], []
            for _ in range(cfg["updates_per_episode"]):
                td_l, spr_l = train_step(net, target_net, optim, replay,
                                         cfg["batch_size"], cfg["gamma"], cfg["spr_coef"])
                td_losses.append(td_l)
                spr_losses.append(spr_l)
            writer.add_scalar("train/td_loss",  float(np.mean(td_losses)),  ep)
            writer.add_scalar("train/spr_loss", float(np.mean(spr_losses)), ep)

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
