"""Stage 5b — MuZero-lite on the afterstate CNN representation.

Three-network architecture (all jointly trained):
  Representation  h: board (20x10) -> latent z  (same CNN encoder as SPR)
  Dynamics        g: (z, action_emb) -> (z', r_hat)  (learned world model)
  Prediction      f: z -> quantile V  (distributional value head)

Training loop (k-step unrolling):
  1. Sample a sequence of k consecutive transitions from replay.
  2. Encode the first board: z_0 = h(s_0)
  3. Unroll k steps via dynamics: z_{t+1} = g(z_t, a_t)
  4. At each step compute:
       - value target:  n-step return  V_t = sum_{i=0}^{k-1} gamma^i r_{t+i} + gamma^k * V_target(z_{t+k})
       - reward target: r_t
       - representation consistency (SPR): cosine(z_t, stop_grad(h_target(s_t)))
  5. Losses: QR-DQN quantile Huber (value) + MSE (reward) + cosine (consistency)

Inference-time planning (latent MCTS):
  For each candidate placement a (enumerated from game):
    1. Encode the afterstate board: z = h(s_after)
    2. For each hypothetical next piece, pick the best next action by V(g(z, a'))
    3. Q(a) = r(z) + gamma * E_piece[max_a' V(g(z, a'))]  -- all in latent space, no deepcopy

Key advantages over Stage 5a (MCTS with deepcopy):
  - Planning is pure tensor ops — no game simulation, ~10-100x faster per node
  - Value function trained with k-step supervision instead of 1-step TD
  - Representation is dynamics-aware (consistency loss forces z to encode future relevance)

Usage:
    python train.py --algo muzero_afterstate
    python train.py --algo muzero_afterstate --config rl_training/configs/muzero_afterstate.json
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
DEFAULT_CONFIG = Path(__file__).parent / "configs" / "muzero_afterstate.json"

BOARD_SHAPE = (Tetris.ROWS, Tetris.COLS)  # (20, 10)
MAX_ROT = 4
MAX_DX = Tetris.COLS + 3  # x in [-3, COLS-1] -> dx in [0, COLS+2]


# ── networks ──────────────────────────────────────────────────────

class MuZeroNet(nn.Module):
    """Representation + dynamics + prediction networks."""

    def __init__(self, conv_channels, fc_hidden, latent_dim,
                 action_emb_dim, n_quantiles, device):
        super().__init__()
        self.n_quantiles = n_quantiles
        R, C = BOARD_SHAPE

        # ── representation: board -> z ────────────────────────────
        conv_layers, in_ch = [], 1
        for ch in conv_channels:
            conv_layers += [nn.Conv2d(in_ch, ch, kernel_size=3, padding=1), nn.ReLU()]
            in_ch = ch
        self.repr_conv = nn.Sequential(*conv_layers)
        flat_dim = (conv_channels[-1] if conv_channels else 1) * R * C
        enc_fc = []
        in_dim = flat_dim
        for h in fc_hidden:
            enc_fc += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        enc_fc.append(nn.Linear(in_dim, latent_dim))
        self.repr_fc = nn.Sequential(*enc_fc)

        # ── action embedding ──────────────────────────────────────
        self.rot_emb = nn.Embedding(MAX_ROT, action_emb_dim)
        self.dx_emb = nn.Embedding(MAX_DX, action_emb_dim)
        act_dim = action_emb_dim * 2

        # ── dynamics: (z, act) -> (z', r_hat) ────────────────────
        # Shared trunk → next latent
        self.dyn_hidden = nn.Sequential(
            nn.Linear(latent_dim + act_dim, latent_dim),
            nn.ReLU(),
        )
        self.dyn_z = nn.Linear(latent_dim, latent_dim)
        # Separate small MLP for reward (avoids conflicting gradients with dyn_z)
        self.dyn_r_hidden = nn.Sequential(
            nn.Linear(latent_dim + act_dim, latent_dim // 2),
            nn.ReLU(),
        )
        self.dyn_r = nn.Linear(latent_dim // 2, 1)

        # ── prediction: z -> scalar V ────────────────────────────────
        # Plain scalar output (true MuZero style). n_quantiles param is kept in
        # config for backwards compat but is no longer used by the network.
        self.value_head = nn.Linear(latent_dim, 1)

        self.to(device)

    def represent(self, x):
        """Board tensor -> latent z. Accepts [R,C], [B,R,C], [B,1,R,C]."""
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32,
                                device=next(self.parameters()).device)
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        feat = self.repr_conv(x).flatten(1)
        return self.repr_fc(feat)   # [B, latent_dim]

    def _action_emb(self, rot, dx):
        device = next(self.parameters()).device
        if not isinstance(rot, torch.Tensor):
            rot = torch.as_tensor(rot, dtype=torch.long, device=device)
        if not isinstance(dx, torch.Tensor):
            dx = torch.as_tensor(dx, dtype=torch.long, device=device)
        return torch.cat([self.rot_emb(rot), self.dx_emb(dx)], dim=-1)  # [B, 2*emb]

    def dynamics(self, z, rot, dx):
        """(z, action) -> (z_next, r_hat_scalar). Returns [B, latent], [B]."""
        act = self._action_emb(rot, dx)
        inp = torch.cat([z, act], dim=-1)
        h = self.dyn_hidden(inp)
        z_next = self.dyn_z(h)
        r_hat = self.dyn_r(self.dyn_r_hidden(inp)).squeeze(-1)
        return z_next, r_hat

    def value(self, z):
        """z -> scalar V [B]."""
        return self.value_head(z).squeeze(-1)

    def forward(self, x):
        """Board -> scalar V. Convenience for greedy action selection."""
        return self.value(self.represent(x))


# ── sequence replay buffer ────────────────────────────────────────

class SequenceReplay:
    """Stores full episodes as lists of (board, rot, dx, reward, done).
    Samples contiguous k-step sequences for unrolled training.

    capacity: maximum number of *transitions* to store (not episodes).
    Old episodes are evicted (FIFO) once the total transition count exceeds capacity.
    """

    def __init__(self, capacity):
        self.capacity = capacity          # transition-level cap
        self.episodes = deque()           # unbounded deque; we evict manually
        self._current = []
        self._total_transitions = 0       # transitions across all stored episodes

    def begin_episode(self):
        self._current = []

    def _commit_episode(self, steps):
        """Add a completed episode to storage and enforce the transition cap."""
        if len(steps) < 2:
            return
        self.episodes.append(list(steps))
        self._total_transitions += len(steps)
        while self._total_transitions > self.capacity and self.episodes:
            evicted = self.episodes.popleft()
            self._total_transitions -= len(evicted)

    def push_step(self, board, rot, dx, reward, done):
        self._current.append((board, rot, dx, reward, done))
        if done:
            self._commit_episode(self._current)
            self._current = []

    def sample_sequences(self, batch_size, k):
        """Sample batch_size sequences of length k+1 (k transitions + terminal V).

        Returns arrays:
          boards [B, k+1, R, C], rots [B, k], dxs [B, k],
          rewards [B, k], dones [B, k]
        """
        R, C = BOARD_SHAPE
        boards = np.zeros((batch_size, k + 1, R, C), dtype=np.float32)
        rots   = np.zeros((batch_size, k), dtype=np.int64)
        dxs    = np.zeros((batch_size, k), dtype=np.int64)
        rewards = np.zeros((batch_size, k), dtype=np.float32)
        dones   = np.zeros((batch_size, k), dtype=np.float32)

        for i in range(batch_size):
            # Pick a random episode long enough to yield k steps
            eligible = [ep for ep in self.episodes if len(ep) > k]
            if not eligible:
                # Fall back to any episode, pad with last step
                ep = random.choice(list(self.episodes))
            else:
                ep = random.choice(eligible)
            # Random start within episode
            max_start = max(0, len(ep) - k - 1)
            start = random.randint(0, max_start)
            for t in range(k + 1):
                idx = min(start + t, len(ep) - 1)
                boards[i, t] = ep[idx][0]
            for t in range(k):
                idx = min(start + t, len(ep) - 1)
                _, rot, dx, reward, done = ep[idx]
                rots[i, t] = rot
                dxs[i, t] = dx
                rewards[i, t] = reward
                dones[i, t] = float(done)

        return boards, rots, dxs, rewards, dones

    def __len__(self):
        return self._total_transitions


# ── quantile Huber loss ───────────────────────────────────────────

def quantile_huber_loss(pred, target, kappa=1.0):
    """QR-DQN loss. pred/target shape [B, N]."""
    n = pred.shape[1]
    device = pred.device
    tau = (torch.arange(n, device=device, dtype=torch.float32) + 0.5) / n
    tau = tau.view(1, n, 1)
    diff = target.unsqueeze(1) - pred.unsqueeze(2)   # [B, N_pred, N_target]
    abs_diff = diff.abs()
    huber = torch.where(abs_diff <= kappa,
                        0.5 * diff.pow(2),
                        kappa * (abs_diff - 0.5 * kappa))
    weight = (tau - (diff.detach() < 0).float()).abs()
    return (weight * huber).sum(dim=1).mean(dim=1).mean()


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
    dx = x + 3
    return action, board, rot, dx


# ── episode runner ────────────────────────────────────────────────

def run_episode(net, eps, replay, max_steps=10000):
    game = Tetris()
    game.next_block()
    total_reward, total_lines, steps = 0.0, 0, 0

    if replay is not None:
        replay.begin_episode()

    while game.state != "gameover" and steps < max_steps:
        action, chosen_board, rot, dx = choose_action(game, net, eps)
        if action is None:
            break
        lines, done = apply_action(game, action)
        reward = nuno_reward(lines, done)
        # Store the afterstate board (post-placement), not the pre-action board.
        # The consistency loss targets h(s_{t+1}) = h(afterstate at step t).
        if replay is not None:
            replay.push_step(chosen_board, rot, dx, reward, done)
        total_reward += reward
        total_lines += lines
        steps += 1
        if done:
            break

    # Flush partial episode (game ended without explicit done flag)
    if replay is not None and replay._current:
        replay._current[-1] = (*replay._current[-1][:4], True)
        replay._commit_episode(replay._current)
        replay._current = []

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

def train_step(net, target_net, optim, replay, cfg):
    batch_size = cfg["batch_size"]
    k = cfg["unroll_steps"]
    gamma = cfg["gamma"]
    reward_coef = cfg["reward_coef"]
    consistency_coef = cfg["consistency_coef"]

    boards, rots, dxs, rewards, dones = replay.sample_sequences(batch_size, k)
    device = next(net.parameters()).device

    boards_t  = torch.as_tensor(boards,  dtype=torch.float32, device=device)  # [B, k+1, R, C]
    rots_t    = torch.as_tensor(rots,    dtype=torch.long,    device=device)  # [B, k]
    dxs_t     = torch.as_tensor(dxs,     dtype=torch.long,    device=device)
    rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device)  # [B, k]
    dones_t   = torch.as_tensor(dones,   dtype=torch.float32, device=device)

    z = net.represent(boards_t[:, 0])   # [B, latent]

    with torch.no_grad():
        z_targets  = [target_net.represent(boards_t[:, t]) for t in range(k + 1)]
        v_targets  = [target_net.value(zt) for zt in z_targets]  # list of [B] scalars

    value_loss = torch.tensor(0.0, device=device)
    reward_loss = torch.tensor(0.0, device=device)
    consistency_loss = torch.tensor(0.0, device=device)

    for t in range(k):
        # ── n-step return target (scalar MSE, true MuZero style) ──
        G = torch.zeros(batch_size, dtype=torch.float32, device=device)
        for i in range(t, k):
            G = G + (gamma ** (i - t)) * rewards_t[:, i] * (1.0 - dones_t[:, i])
        mask = torch.ones(batch_size, device=device)
        for i in range(t, k):
            mask = mask * (1.0 - dones_t[:, i])
        G = G + (gamma ** (k - t)) * v_targets[k] * mask

        v_pred = net.value(z)   # [B]
        value_loss = value_loss + F.mse_loss(v_pred, G)

        # ── dynamics step ─────────────────────────────────────────
        z_next, r_hat = net.dynamics(z, rots_t[:, t], dxs_t[:, t])

        # ── reward prediction loss ────────────────────────────────
        reward_loss = reward_loss + F.mse_loss(r_hat, rewards_t[:, t])

        # ── representation consistency (SPR) ─────────────────────
        consistency_loss = consistency_loss + (
            1.0 - F.cosine_similarity(z_next, z_targets[t + 1], dim=-1).mean()
        )

        z = z_next  # unroll

    loss = (value_loss / k
            + reward_coef * reward_loss / k
            + consistency_coef * consistency_loss / k)

    optim.zero_grad()
    loss.backward()
    grad_clip = cfg.get("grad_clip", 0.0)
    if grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
    optim.step()
    return (float(value_loss.item() / k),
            float(reward_loss.item() / k),
            float(consistency_loss.item() / k))


# ── latent-space MCTS planning (inference only) ───────────────────

ALL_PIECE_TYPES = list(range(7))


def muzero_action(game, net, depth=1, gamma=0.95):
    """Choose action using latent-space expectimax.

    Unlike Stage 5a's MCTS (which deepcopies the game), this unrolls
    purely in latent space using the learned dynamics net. No simulation.

    Args:
        game: current Tetris game (read-only).
        net: MuZeroNet instance.
        depth: lookahead depth (1 = one-step expectation over next piece).
        gamma: discount.

    Returns (action, board) where action = (rot, x), or (None, None).
    """
    from tetris.env.afterstate import enumerate_afterstates_raw

    afterstates = enumerate_afterstates_raw(game)
    if not afterstates:
        return None, None

    actions = list(afterstates.keys())
    device = next(net.parameters()).device

    with torch.no_grad():
        q_values = _latent_expectimax(afterstates, net, depth, gamma, device)

    if not q_values:
        return None, None

    best_action = max(q_values, key=lambda a: q_values[a])
    return best_action, afterstates[best_action][0]


def _latent_expectimax(afterstates, net, depth, gamma, device):
    """Expectimax in latent space. Returns {action: Q_value}.

    depth=1: Q(a) = r_hat(z_a) + gamma * V(z_a)
    depth>1: unroll dynamics greedily for (depth-1) more steps in latent space,
             then bootstrap with V at the leaf. No game simulation needed.
    """
    # Batch-encode all afterstate boards
    actions = list(afterstates.keys())
    boards = np.stack([afterstates[a][0] for a in actions])   # [A, R, C]
    boards_t = torch.as_tensor(boards, dtype=torch.float32, device=device)
    z_all = net.represent(boards_t)   # [A, latent]

    q_values = {}
    for i, action in enumerate(actions):
        rot, x = action
        dx = x + 3
        z = z_all[i:i+1]   # [1, latent]

        # Use dynamics to get predicted reward and transition
        rot_t = torch.tensor([rot % MAX_ROT], dtype=torch.long, device=device)
        dx_t  = torch.tensor([dx],            dtype=torch.long, device=device)
        z_next, r_hat = net.dynamics(z, rot_t, dx_t)
        total_r = float(r_hat.item())

        if depth <= 1:
            leaf_v = float(net.value(z_next).item())
            q_values[action] = total_r + gamma * leaf_v
        else:
            # Greedily unroll (depth-1) more steps: at each step take argmax V
            # over a fixed dummy action (same action repeated) — an approximation
            # since we don't re-enumerate placements. Practical depth is 1 or 2.
            z_cur = z_next
            disc = gamma
            for _ in range(depth - 1):
                z_cur, r_step = net.dynamics(z_cur, rot_t, dx_t)
                total_r += disc * float(r_step.item())
                disc *= gamma
            leaf_v = float(net.value(z_cur).item())
            q_values[action] = total_r + disc * leaf_v

    return q_values


# ── config ────────────────────────────────────────────────────────

def load_config(path=None):
    with open(DEFAULT_CONFIG) as f:
        cfg = json.load(f)
    if path is not None:
        with open(path) as f:
            cfg.update(json.load(f))
    return cfg


def _build_net(cfg, device):
    return MuZeroNet(
        cfg["conv_channels"],
        cfg["fc_hidden"],
        cfg["latent_dim"],
        cfg["action_emb_dim"],
        cfg["n_quantiles"],
        device,
    )


def run_trial(cfg, trial_episodes):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = _build_net(cfg, device)
    target_net = _build_net(cfg, device)
    target_net.load_state_dict(net.state_dict())
    optim = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    replay = SequenceReplay(cfg["buffer_capacity"])

    eps = cfg["eps_start"]
    eps_end = cfg["eps_end"]
    decay = (eps - eps_end) / max(cfg["eps_decay_episodes"], 1)
    best = -float("inf")

    for ep in range(trial_episodes):
        run_episode(net, eps, replay)
        if len(replay) >= cfg["min_replay_size"]:
            for _ in range(cfg["updates_per_episode"]):
                train_step(net, target_net, optim, replay, cfg)
        tau = cfg.get("ema_tau", 0.0)
        if tau > 0.0:
            with torch.no_grad():
                for p, pt in zip(net.parameters(), target_net.parameters()):
                    pt.data.mul_(1.0 - tau).add_(tau * p.data)
        elif ep % cfg["target_update_freq"] == 0:
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"MuZero Afterstate config: {json.dumps(cfg, indent=2)}")

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = LOGS_DIR / "muzero_afterstate" / run_name
    weights_dir = WEIGHTS_DIR / "muzero_afterstate" / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    for d in (log_dir, weights_dir):
        with open(d / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)
    writer = SummaryWriter(str(log_dir))

    net = _build_net(cfg, device)
    target_net = _build_net(cfg, device)
    target_net.load_state_dict(net.state_dict())
    optim = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    replay = SequenceReplay(cfg["buffer_capacity"])

    eps = cfg["eps_start"]
    eps_end = cfg["eps_end"]
    decay = (eps - eps_end) / max(cfg["eps_decay_episodes"], 1)
    best_eval = -float("inf")
    total_transitions = 0

    for ep in range(int(cfg["episodes"])):
        ep_reward, ep_steps, ep_lines = run_episode(net, eps, replay)
        total_transitions += ep_steps
        writer.add_scalar("train/reward",      ep_reward,         ep)
        writer.add_scalar("train/lines",       ep_lines,          ep)
        writer.add_scalar("train/steps",       ep_steps,          ep)
        writer.add_scalar("train/eps",         eps,               ep)
        writer.add_scalar("train/transitions", total_transitions, ep)

        if len(replay) >= cfg["min_replay_size"]:
            v_losses, r_losses, c_losses = [], [], []
            for _ in range(cfg["updates_per_episode"]):
                vl, rl, cl = train_step(net, target_net, optim, replay, cfg)
                v_losses.append(vl); r_losses.append(rl); c_losses.append(cl)
            writer.add_scalar("train/value_loss",       float(np.mean(v_losses)), ep)
            writer.add_scalar("train/reward_loss",      float(np.mean(r_losses)), ep)
            writer.add_scalar("train/consistency_loss", float(np.mean(c_losses)), ep)

        # Soft EMA target update (Polyak): θ_target ← τ·θ + (1-τ)·θ_target
        tau = cfg.get("ema_tau", 0.0)
        if tau > 0.0:
            with torch.no_grad():
                for p, pt in zip(net.parameters(), target_net.parameters()):
                    pt.data.mul_(1.0 - tau).add_(tau * p.data)
        elif ep % cfg["target_update_freq"] == 0:
            target_net.load_state_dict(net.state_dict())

        if eps > eps_end:
            eps = max(eps_end, eps - decay)

        if (ep + 1) % cfg["eval_freq"] == 0:
            mean_r, mean_lines, mean_steps = evaluate(net, cfg["eval_episodes"])
            writer.add_scalar("eval/reward",            mean_r,            ep)
            writer.add_scalar("eval/lines",             mean_lines,        ep)
            writer.add_scalar("eval/steps",             mean_steps,        ep)
            # Also log against transition count for fair cross-run comparison
            writer.add_scalar("eval_by_transition/reward", mean_r,     total_transitions)
            writer.add_scalar("eval_by_transition/lines",  mean_lines, total_transitions)
            print(f"ep {ep+1}/{cfg['episodes']} | eval reward {mean_r:.2f} | "
                  f"lines {mean_lines:.1f} | steps {mean_steps:.0f} | "
                  f"transitions {total_transitions} | eps {eps:.3f}")
            if mean_r > best_eval:
                best_eval = mean_r
                torch.save(net.state_dict(), weights_dir / "best.pth")

        if (ep + 1) % cfg["checkpoint_freq"] == 0:
            torch.save(net.state_dict(), weights_dir / f"ep{ep+1:04d}.pth")

    writer.close()
    print(f"\nDone. Best eval reward: {best_eval:.2f} | Total transitions: {total_transitions}")
    return best_eval


if __name__ == "__main__":
    main()
