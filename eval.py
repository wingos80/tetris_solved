"""Evaluate a trained Tetris RL policy — render, GIF, and/or benchmark stats.

Supports all algos (tianshou-based and afterstate variants).

Single-episode mode (default):  renders live + saves a GIF.
Benchmark mode (--episodes N):  runs N greedy episodes, prints mean±std stats.

Usage:
    python eval.py --algo afterstate_qrdqn                 # render 1 ep, save GIF
    python eval.py --algo afterstate_qrdqn --no-gif        # render only
    python eval.py --algo afterstate_qrdqn --episodes 30   # 30-ep benchmark, no render
    python eval.py --algo sac_v2 --episodes 30             # benchmark any algo
    python eval.py path/to/best.pth --algo ppo             # explicit weights

Flags:
    path            Optional explicit path to .pth file
    --algo          Algorithm (required for correct network architecture)
    --episodes N    Episodes to run; default 1 (renders). N>1 disables render/GIF.
    --render        Force rendering even in benchmark mode
    --no-gif        Skip saving GIF in single-episode mode
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "rl_training"))

import numpy as np
import torch

from tetris.env import TetrisEnv
from tetris.env.afterstate import (
    apply_action,
    enumerate_afterstates,
    enumerate_afterstates_raw,
    nuno_reward,
)
from tetris.game import Tetris

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HIDDEN_2 = [256, 256]
HIDDEN_3 = [256, 256, 256]
OBS_N, ACT_N = 78, 40


# ── tianshou policy builders ──────────────────────────────────────

def build_ppo_policy():
    from tianshou.policy import PPOPolicy
    from tianshou.utils.net.common import Net
    from tianshou.utils.net.discrete import Actor, Critic
    from torch.distributions import Categorical
    from ppo import MaskedActor, ObsExtractCritic

    net_a = Net(state_shape=OBS_N, hidden_sizes=HIDDEN_2, device=DEVICE)
    net_c = Net(state_shape=OBS_N, hidden_sizes=HIDDEN_2, device=DEVICE)
    raw_actor = Actor(net_a, ACT_N, device=DEVICE, softmax_output=False).to(DEVICE)
    raw_critic = Critic(net_c, device=DEVICE).to(DEVICE)
    actor = MaskedActor(raw_actor).to(DEVICE)
    critic = ObsExtractCritic(raw_critic).to(DEVICE)
    policy = PPOPolicy(
        actor, critic, torch.optim.Adam(actor.parameters()),
        dist_fn=lambda logits: Categorical(logits=logits),
        action_space=TetrisEnv().action_space,
    )
    return policy


def build_rainbow_policy():
    from tianshou.policy import RainbowPolicy
    from rainbow import RainbowNet, NUM_ATOMS, V_MIN, V_MAX

    net = RainbowNet(OBS_N, ACT_N, HIDDEN_2[0], NUM_ATOMS, DEVICE)
    policy = RainbowPolicy(
        net, torch.optim.Adam(net.parameters()),
        num_atoms=NUM_ATOMS, v_min=V_MIN, v_max=V_MAX,
        estimation_step=3, target_update_freq=500,
    )
    return policy


def build_sac_policy():
    from tianshou.policy import DiscreteSACPolicy
    from tianshou.utils.net.common import Net
    from sac import MaskedSACActor, ObsExtractCritic

    net_a = Net(state_shape=OBS_N, hidden_sizes=HIDDEN_2, device=DEVICE)
    net_c1 = Net(state_shape=OBS_N, hidden_sizes=HIDDEN_2, device=DEVICE)
    net_c2 = Net(state_shape=OBS_N, hidden_sizes=HIDDEN_2, device=DEVICE)
    actor = MaskedSACActor(net_a, ACT_N, DEVICE).to(DEVICE)
    critic1 = ObsExtractCritic(net_c1, ACT_N, DEVICE).to(DEVICE)
    critic2 = ObsExtractCritic(net_c2, ACT_N, DEVICE).to(DEVICE)
    policy = DiscreteSACPolicy(
        actor, torch.optim.Adam(actor.parameters()),
        critic1, torch.optim.Adam(critic1.parameters()),
        critic2, torch.optim.Adam(critic2.parameters()),
        tau=0.005, gamma=0.99, alpha=0.2, estimation_step=3,
    )
    return policy


def build_sac_v2_policy(weights_path=None):
    from tianshou.policy import DiscreteSACPolicy
    from sac_v2 import MaskedSACActor, ObsExtractCritic, _make_net

    cfg = {"spectral_norm": False, "layer_norm": False}
    if weights_path:
        config_path = Path(weights_path).parent / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg.update(json.load(f))

    hidden = cfg.get("hidden", HIDDEN_3)
    net_a = _make_net(OBS_N, hidden, cfg)
    net_c1 = _make_net(OBS_N, hidden, cfg)
    net_c2 = _make_net(OBS_N, hidden, cfg)
    actor = MaskedSACActor(net_a, ACT_N, DEVICE).to(DEVICE)
    critic1 = ObsExtractCritic(net_c1, ACT_N, DEVICE).to(DEVICE)
    critic2 = ObsExtractCritic(net_c2, ACT_N, DEVICE).to(DEVICE)
    policy = DiscreteSACPolicy(
        actor, torch.optim.Adam(actor.parameters()),
        critic1, torch.optim.Adam(critic1.parameters()),
        critic2, torch.optim.Adam(critic2.parameters()),
        tau=0.005, gamma=0.99, alpha=0.2, estimation_step=5,
    )
    return policy


def build_dqn_policy():
    from tianshou.policy import DQNPolicy
    from dqn import QNet

    net = QNet(OBS_N, ACT_N, HIDDEN_3, DEVICE)
    policy = DQNPolicy(
        net, torch.optim.Adam(net.parameters()),
        discount_factor=0.99, estimation_step=3,
        target_update_freq=500, is_double=True,
    ).to(DEVICE)
    policy.set_eps(0.0)
    return policy


def build_qrdqn_policy(weights_path=None):
    from tianshou.policy import QRDQNPolicy
    from qrdqn import QRNet

    num_quantiles = 51
    if weights_path:
        config_path = Path(weights_path).parent / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                num_quantiles = json.load(f).get("num_quantiles", 51)
    net = QRNet(OBS_N, ACT_N, HIDDEN_3, num_quantiles, DEVICE)
    policy = QRDQNPolicy(
        net, torch.optim.Adam(net.parameters()),
        discount_factor=0.99, num_quantiles=num_quantiles,
        estimation_step=3, target_update_freq=500,
    ).to(DEVICE)
    policy.set_eps(0.0)
    return policy


def build_iqn_policy(weights_path=None):
    from tianshou.policy import IQNPolicy
    from iqn import ObsExtractNet
    from tianshou.utils.net.discrete import ImplicitQuantileNetwork

    cfg = {}
    if weights_path:
        config_path = Path(weights_path).parent / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)

    preprocess = ObsExtractNet(OBS_N, cfg.get("hidden", HIDDEN_3), DEVICE)
    net = ImplicitQuantileNetwork(
        preprocess, ACT_N,
        num_cosines=int(cfg.get("num_cosines", 64)),
        preprocess_net_output_dim=preprocess.output_dim,
        device=DEVICE,
    )
    policy = IQNPolicy(
        net, torch.optim.Adam(net.parameters()),
        discount_factor=0.99,
        sample_size=int(cfg.get("sample_size", 32)),
        online_sample_size=int(cfg.get("online_sample_size", 8)),
        target_sample_size=int(cfg.get("target_sample_size", 8)),
        estimation_step=3, target_update_freq=500,
    ).to(DEVICE)
    policy.set_eps(0.0)
    return policy


BUILDERS = {
    "ppo": build_ppo_policy,
    "rainbow": build_rainbow_policy,
    "sac": build_sac_policy,
    "sac_v2": build_sac_v2_policy,
    "dqn": build_dqn_policy,
    "qrdqn": build_qrdqn_policy,
    "iqn": build_iqn_policy,
}

_PATH_AWARE_BUILDERS = {"sac_v2", "qrdqn", "iqn"}

AFTERSTATE_ALGOS = {"afterstate", "afterstate_qrdqn", "afterstate_cnn", "afterstate_spr", "muzero_afterstate"}
MCTS_ELIGIBLE = {"afterstate", "afterstate_qrdqn"}  # 4-feature V-nets only


# ── helpers ───────────────────────────────────────────────────────

def _latest_weights(algo):
    algo_dir = ROOT / "rl_training" / "weights" / algo
    if not algo_dir.is_dir():
        return None
    runs = sorted(
        [d for d in algo_dir.iterdir() if d.is_dir() and (d / "best.pth").exists()],
        key=lambda d: d.name, reverse=True,
    )
    return str(runs[0] / "best.pth") if runs else None


def load_policy(path, algo):
    if algo in _PATH_AWARE_BUILDERS:
        policy = BUILDERS[algo](weights_path=path)
    else:
        policy = BUILDERS[algo]()
    policy.load_state_dict(torch.load(path, map_location=DEVICE))
    policy.to(DEVICE).eval()
    return policy


def get_action(policy, obs, algo):
    device = next(policy.parameters()).device
    obs_t = torch.as_tensor(obs["obs"], dtype=torch.float32, device=device).unsqueeze(0)
    mask_t = torch.as_tensor(obs["mask"], dtype=torch.bool, device=device).unsqueeze(0)

    with torch.no_grad():
        if algo in ("rainbow", "dqn", "qrdqn", "iqn"):
            from tianshou.data import Batch
            mask_int = mask_t.to(torch.int8)
            batch = Batch(obs=Batch(obs=obs_t, mask=mask_int), info={})
            result = policy(batch)
            return result.act.item()
        else:
            class Obs:
                pass
            o = Obs()
            o.obs = obs_t
            o.mask = mask_t
            logits, _ = policy.actor(o)
            return logits.argmax(dim=-1).item()


def _load_afterstate_net(path, algo):
    """Return (net, score_fn, enumerator) for an afterstate algo."""
    cfg_path = Path(path).parent / "config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    if algo == "afterstate_qrdqn":
        from rl_training.afterstate_qrdqn import QuantileVNet
        net = QuantileVNet(cfg["hidden"], cfg["n_quantiles"], DEVICE)
        score_fn = lambda x: net.value(x).cpu().numpy()
        enumerator = enumerate_afterstates
    elif algo == "afterstate_cnn":
        from rl_training.afterstate_cnn import BoardVNet
        net = BoardVNet(cfg["conv_channels"], cfg["fc_hidden"], DEVICE)
        score_fn = lambda x: net(x).cpu().numpy()
        enumerator = enumerate_afterstates_raw
    elif algo == "afterstate_spr":
        from rl_training.afterstate_spr import SPRNet
        net = SPRNet(cfg["conv_channels"], cfg["fc_hidden"],
                     cfg["latent_dim"], cfg["action_emb_dim"], DEVICE)
        score_fn = lambda x: net(x).cpu().numpy()
        enumerator = enumerate_afterstates_raw
    elif algo == "muzero_afterstate":
        from rl_training.muzero_afterstate import MuZeroNet
        net = MuZeroNet(cfg["conv_channels"], cfg["fc_hidden"],
                        cfg["latent_dim"], cfg["action_emb_dim"],
                        cfg["n_quantiles"], DEVICE)
        score_fn = lambda x: net(x).cpu().numpy()
        enumerator = enumerate_afterstates_raw
    else:
        from rl_training.afterstate_dqn import VNet
        net = VNet(cfg["hidden"], DEVICE)
        score_fn = lambda x: net(x).cpu().numpy()
        enumerator = enumerate_afterstates
    net.load_state_dict(torch.load(path, map_location=DEVICE))
    net.eval()
    return net, score_fn, enumerator


# ── episode runners ───────────────────────────────────────────────

def _run_tianshou_episode(policy, algo, render, save_gif, gif_path):
    """One episode with a tianshou policy. Returns (reward, lines, placements)."""
    import pygame
    from PIL import Image
    from tetris.renderer import TetrisRenderer

    env = TetrisEnv()
    obs, _ = env.reset()
    game = env.game

    if render:
        pygame.init()
        renderer = TetrisRenderer(game)

    frames = []
    clock = pygame.time.Clock() if render else None
    ep_reward, ep_lines, ep_placements = 0.0, 0, 0

    while True:
        action = get_action(policy, obs, algo)
        obs, reward, term, trunc, info = env.step(action)
        ep_reward += reward
        if info.get("valid", False):
            ep_placements += 1
            ep_lines += info.get("lines", 0)

        if render:
            renderer.draw()
            pygame.display.flip()
            clock.tick(5)
            if save_gif:
                data = pygame.image.tostring(renderer.screen, "RGB")
                w, h = renderer.screen.get_size()
                frames.append(Image.frombytes("RGB", (w, h), data))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

        if term or trunc or ep_placements > 5000:
            break

    if save_gif and frames:
        os.makedirs(os.path.dirname(gif_path) or ".", exist_ok=True)
        frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                       duration=200, loop=0)
        print(f"Saved GIF → {gif_path} ({len(frames)} frames)")

    if render:
        pygame.quit()

    return ep_reward, ep_lines, ep_placements


def _run_afterstate_episode(score_fn, enumerator, render, save_gif, gif_path):
    """One episode with an afterstate net. Returns (reward, lines, placements)."""
    import pygame
    from PIL import Image
    from tetris.renderer import TetrisRenderer

    game = Tetris()
    game.next_block()

    if render:
        pygame.init()
        renderer = TetrisRenderer(game)

    frames = []
    clock = pygame.time.Clock() if render else None
    ep_reward, ep_lines, ep_placements = 0.0, 0, 0

    while game.state != "gameover":
        afterstates = enumerator(game)
        if not afterstates:
            break
        actions = list(afterstates.keys())
        feats = np.stack([afterstates[a][0] for a in actions])
        with torch.no_grad():
            values = score_fn(feats)
        action = actions[int(np.argmax(values))]
        lines, done = apply_action(game, action)
        ep_reward += nuno_reward(lines, done)
        ep_lines += lines
        ep_placements += 1

        if render:
            renderer.draw()
            pygame.display.flip()
            clock.tick(5)
            if save_gif:
                data = pygame.image.tostring(renderer.screen, "RGB")
                w, h = renderer.screen.get_size()
                frames.append(Image.frombytes("RGB", (w, h), data))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

        if done or ep_placements > 5000:
            break

    if save_gif and frames:
        os.makedirs(os.path.dirname(gif_path) or ".", exist_ok=True)
        frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                       duration=200, loop=0)
        print(f"Saved GIF → {gif_path} ({len(frames)} frames)")

    if render:
        pygame.quit()

    return ep_reward, ep_lines, ep_placements


def _run_mcts_episode(net_value_fn, depth, gamma, render, save_gif, gif_path):
    """One episode using MCTS expectimax. Returns (reward, lines, placements)."""
    from rl_training.mcts import mcts_action
    import pygame
    from PIL import Image
    from tetris.renderer import TetrisRenderer

    game = Tetris()
    game.next_block()

    if render:
        pygame.init()
        renderer = TetrisRenderer(game)

    frames = []
    clock = pygame.time.Clock() if render else None
    ep_reward, ep_lines, ep_placements = 0.0, 0, 0

    while game.state != "gameover":
        action, _ = mcts_action(game, net_value_fn, depth=depth, gamma=gamma)
        if action is None:
            break
        lines, done = apply_action(game, action)
        ep_reward += nuno_reward(lines, done)
        ep_lines += lines
        ep_placements += 1

        if render:
            renderer.draw()
            pygame.display.flip()
            clock.tick(5)
            if save_gif:
                data = pygame.image.tostring(renderer.screen, "RGB")
                w, h = renderer.screen.get_size()
                frames.append(Image.frombytes("RGB", (w, h), data))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

        if done or ep_placements > 5000:
            break

    if save_gif and frames:
        os.makedirs(os.path.dirname(gif_path) or ".", exist_ok=True)
        frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                       duration=200, loop=0)
        print(f"Saved GIF → {gif_path} ({len(frames)} frames)")

    if render:
        pygame.quit()

    return ep_reward, ep_lines, ep_placements


# ── benchmark helpers (kept for test compatibility) ───────────────

def eval_tianshou_agent(path, algo, episodes):
    """N greedy episodes for a tianshou policy. Returns (rewards, lines, placements)."""
    policy = load_policy(path, algo)
    rewards, lines_list, placements = [], [], []
    for _ in range(episodes):
        r, l, p = _run_tianshou_episode(policy, algo, render=False, save_gif=False, gif_path=None)
        rewards.append(r)
        lines_list.append(l)
        placements.append(p)
    return rewards, lines_list, placements


def eval_afterstate_agent(path, episodes, algo="afterstate"):
    """N greedy episodes for an afterstate net. Returns (rewards, lines, placements)."""
    _, score_fn, enumerator = _load_afterstate_net(path, algo)
    rewards, lines_list, placements = [], [], []
    for _ in range(episodes):
        r, l, p = _run_afterstate_episode(score_fn, enumerator, render=False, save_gif=False, gif_path=None)
        rewards.append(r)
        lines_list.append(l)
        placements.append(p)
    return rewards, lines_list, placements


# ── main ──────────────────────────────────────────────────────────

ALL_ALGOS = list(BUILDERS) + list(AFTERSTATE_ALGOS) + ["mcts"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a Tetris RL policy")
    parser.add_argument("path", nargs="?", default=None,
                        help="Path to .pth weights file (default: latest for --algo)")
    parser.add_argument("--algo", choices=ALL_ALGOS, default="ppo")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Episodes to run (default 1 = render+GIF; >1 = benchmark)")
    parser.add_argument("--render", action="store_true",
                        help="Force rendering even in benchmark mode")
    parser.add_argument("--no-gif", action="store_true",
                        help="Skip saving GIF in single-episode mode")
    parser.add_argument("--mcts-base", choices=list(MCTS_ELIGIBLE), default="afterstate_qrdqn",
                        help="V-net to use as MCTS leaf evaluator (default: afterstate_qrdqn)")
    parser.add_argument("--mcts-depth", type=int, default=1,
                        help="MCTS expectimax depth (default: 1)")
    parser.add_argument("--gamma", type=float, default=0.95,
                        help="Discount for MCTS backup (default: 0.95)")
    args = parser.parse_args()

    algo_for_weights = args.mcts_base if args.algo == "mcts" else args.algo

    if args.path is None:
        args.path = _latest_weights(algo_for_weights)
        if args.path is None:
            print(f"No weights found for {algo_for_weights}. Pass an explicit path.")
            sys.exit(1)
        print(f"Using latest weights: {args.path}")

    render = (args.episodes == 1) or args.render
    save_gif = render and (args.episodes == 1) and not args.no_gif
    gif_path = str(ROOT / "rl_training" / "replays" / f"best_{args.algo}.gif")

    rewards, lines_list, placements = [], [], []

    if args.algo == "mcts":
        _, net_value_fn, _ = _load_afterstate_net(args.path, args.mcts_base)
        # Wrap score_fn to match mcts_action's expected signature: (features [N,4]) -> [N]
        def _value_fn(x):
            with torch.no_grad():
                return net_value_fn(x)
        label = f"mcts(depth={args.mcts_depth}, base={args.mcts_base})"
        print(f"MCTS: depth={args.mcts_depth}, base={args.mcts_base}, gamma={args.gamma}")
        for ep in range(args.episodes):
            r, l, p = _run_mcts_episode(
                _value_fn, depth=args.mcts_depth, gamma=args.gamma,
                render=render and (ep == 0 or args.render),
                save_gif=save_gif and ep == 0,
                gif_path=gif_path,
            )
            rewards.append(r); lines_list.append(l); placements.append(p)
            print(f"ep {ep+1}/{args.episodes} | reward={r:.2f}  lines={l}  placements={p}")
    elif args.algo in AFTERSTATE_ALGOS:
        _, score_fn, enumerator = _load_afterstate_net(args.path, args.algo)
        for ep in range(args.episodes):
            r, l, p = _run_afterstate_episode(
                score_fn, enumerator,
                render=render and (ep == 0 or args.render),
                save_gif=save_gif and ep == 0,
                gif_path=gif_path,
            )
            rewards.append(r); lines_list.append(l); placements.append(p)
            print(f"ep {ep+1}/{args.episodes} | reward={r:.2f}  lines={l}  placements={p}")
    else:
        policy = load_policy(args.path, args.algo)
        for ep in range(args.episodes):
            r, l, p = _run_tianshou_episode(
                policy, args.algo,
                render=render and (ep == 0 or args.render),
                save_gif=save_gif and ep == 0,
                gif_path=gif_path,
            )
            rewards.append(r); lines_list.append(l); placements.append(p)
            print(f"ep {ep+1}/{args.episodes} | reward={r:.2f}  lines={l}  placements={p}")

    if args.episodes > 1:
        r = np.array(rewards); l = np.array(lines_list); p = np.array(placements)
        display_algo = f"mcts(d={args.mcts_depth},{args.mcts_base})" if args.algo == "mcts" else args.algo
        print(f"\n{'='*50}")
        print(f"Algo: {display_algo}  |  Weights: {args.path}")
        print(f"Episodes: {args.episodes}")
        print(f"{'='*50}")
        print(f"Reward:     {r.mean():>8.2f} ± {r.std():.2f}   [min {r.min():.2f}, max {r.max():.2f}]")
        print(f"Lines:      {l.mean():>8.2f} ± {l.std():.2f}   [min {l.min()}, max {l.max()}]")
        print(f"Placements: {p.mean():>8.2f} ± {p.std():.2f}   [min {p.min()}, max {p.max()}]")
