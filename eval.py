"""Evaluate a trained Tetris RL policy — render with pygame and save a GIF.

Loads saved weights from rl_training/weights/, runs one episode, and saves
an animated GIF to rl_training/replays/best_{algo}.gif.

Supported algos and their weight files:
    ppo     → rl_training/weights/best_policy.pth
    rainbow → rl_training/weights/best_rainbow.pth
    sac     → rl_training/weights/best_sac.pth
    sac_v2  → rl_training/weights/best_sac_v2.pth
    dqn     → rl_training/weights/best_dqn.pth

Usage:
    python eval.py                               # defaults: ppo
    python eval.py --algo rainbow
    python eval.py --algo sac
    python eval.py --algo sac_v2
    python eval.py --algo dqn
    python eval.py rl_training/weights/best_policy.pth --algo ppo --no-gif

Flags:
    path        Optional explicit path to .pth weights file
    --algo      Algorithm architecture to load (required to build correct network)
    --no-gif    Skip saving the GIF (render only)
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "rl_training"))  # allow 'from ppo import ...' etc.

import torch
import numpy as np
import pygame
from PIL import Image

from tetris.env import TetrisEnv
from tetris.renderer import TetrisRenderer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HIDDEN_2 = [256, 256]
HIDDEN_3 = [256, 256, 256]
OBS_N, ACT_N = 78, 40


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

    # Load config from weights dir if available
    cfg = {"spectral_norm": False, "layer_norm": False}
    if weights_path:
        config_path = Path(weights_path).parent / "config.json"
        if config_path.exists():
            import json
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
            import json
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
            import json
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

# Builders that accept weights_path kwarg (need config.json for arch detection)
_PATH_AWARE_BUILDERS = {"sac_v2", "qrdqn", "iqn"}


def load_policy(path, algo):
    if algo in _PATH_AWARE_BUILDERS:
        policy = BUILDERS[algo](weights_path=path)
    else:
        policy = BUILDERS[algo]()
    policy.load_state_dict(torch.load(path, map_location=DEVICE))
    policy.to(DEVICE)
    policy.eval()
    return policy


def get_action(policy, obs, algo):
    """Get greedy action from policy."""
    device = next(policy.parameters()).device
    obs_t = torch.as_tensor(obs["obs"], dtype=torch.float32, device=device).unsqueeze(0)
    mask_t = torch.as_tensor(obs["mask"], dtype=torch.bool, device=device).unsqueeze(0)

    with torch.no_grad():
        if algo in ("rainbow", "dqn", "qrdqn", "iqn"):
            # DQN-family: use policy forward (handles masking internally)
            # tianshou's compute_q_value does `1 - mask`, so mask must be numeric
            from tianshou.data import Batch
            mask_int = mask_t.to(torch.int8)
            batch = Batch(obs=Batch(obs=obs_t, mask=mask_int), info={})
            result = policy(batch)
            return result.act.item()
        else:
            # PPO / SAC / SAC v2: actor outputs logits, take argmax
            class Obs:
                pass
            o = Obs()
            o.obs = obs_t
            o.mask = mask_t
            logits, _ = policy.actor(o)
            return logits.argmax(dim=-1).item()


def run_episode(policy, algo, render=True, save_gif=True, gif_path=None):
    if gif_path is None:
        gif_path = str(ROOT / "rl_training" / "replays" / "best.gif")
    env = TetrisEnv()
    obs, _ = env.reset()
    game = env.game

    if render:
        pygame.init()
        renderer = TetrisRenderer(game)

    frames = []
    clock = pygame.time.Clock() if render else None
    total_reward = 0
    steps = 0

    while True:
        action = get_action(policy, obs, algo)
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        steps += 1

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

        if term:
            break

    if save_gif and frames:
        os.makedirs(os.path.dirname(gif_path) or ".", exist_ok=True)
        frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                       duration=200, loop=0)
        print(f"Saved GIF to {gif_path} ({len(frames)} frames)")

    if render:
        pygame.quit()

    print(f"Episode: algo={algo}, score={info.get('score', 0)}, "
          f"reward={total_reward:.2f}, steps={steps}")
    return total_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default=None)
    parser.add_argument("--algo", choices=["ppo", "rainbow", "sac", "sac_v2", "dqn", "qrdqn", "iqn"], default="ppo")
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()

    if args.path is None:
        # Find the latest timestamped run dir for this algo
        algo_weights_dir = ROOT / "rl_training" / "weights" / args.algo
        if algo_weights_dir.is_dir():
            run_dirs = sorted(
                [d for d in algo_weights_dir.iterdir()
                 if d.is_dir() and (d / "best.pth").exists()],
                key=lambda d: d.name, reverse=True,
            )
        else:
            run_dirs = []
        if run_dirs:
            args.path = str(run_dirs[0] / "best.pth")
            print(f"Using latest weights: {args.path}")
        else:
            print(f"No weights found in weights/{args.algo}/ — pass an explicit path.")
            sys.exit(1)

    gif_name = str(ROOT / "rl_training" / "replays" / f"best_{args.algo}.gif")

    policy = load_policy(args.path, args.algo)
    run_episode(policy, args.algo, render=True,
                save_gif=not args.no_gif, gif_path=gif_name)
