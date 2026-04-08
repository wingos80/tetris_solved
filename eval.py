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


def build_sac_v2_policy():
    from tianshou.policy import DiscreteSACPolicy
    from tianshou.utils.net.common import Net
    from sac_v2 import MaskedSACActor, ObsExtractCritic

    net_a = Net(state_shape=OBS_N, hidden_sizes=HIDDEN_3, device=DEVICE)
    net_c1 = Net(state_shape=OBS_N, hidden_sizes=HIDDEN_3, device=DEVICE)
    net_c2 = Net(state_shape=OBS_N, hidden_sizes=HIDDEN_3, device=DEVICE)
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


def build_qrdqn_policy():
    from tianshou.policy import QRDQNPolicy
    from qrdqn import QRNet

    num_quantiles = 200
    net = QRNet(OBS_N, ACT_N, HIDDEN_3, num_quantiles, DEVICE)
    policy = QRDQNPolicy(
        net, torch.optim.Adam(net.parameters()),
        discount_factor=0.99, num_quantiles=num_quantiles,
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
}


def load_policy(path, algo):
    policy = BUILDERS[algo]()
    policy.load_state_dict(torch.load(path, map_location=DEVICE))
    policy.to(DEVICE)
    policy.eval()
    return policy


def get_action(policy, obs, algo):
    """Get greedy action from policy."""
    obs_t = torch.as_tensor(obs["obs"], dtype=torch.float32, device=DEVICE).unsqueeze(0)
    mask_t = torch.as_tensor(obs["mask"], dtype=torch.bool, device=DEVICE).unsqueeze(0)

    with torch.no_grad():
        if algo == "rainbow":
            # Rainbow: DQN-family, mask must be int8 (tianshou does 1-mask, breaks on bool)
            from tianshou.data import Batch
            mask_int = torch.as_tensor(obs["mask"], dtype=torch.int8, device=DEVICE).unsqueeze(0)
            batch = Batch(obs=Batch(obs=obs_t, mask=mask_int), info={})
            result = policy(batch)
            return result.act.item()
        else:
            # PPO / SAC: actor outputs logits, take argmax
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
    parser.add_argument("--algo", choices=["ppo", "rainbow", "sac", "sac_v2", "dqn", "qrdqn"], default="ppo")
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()

    _weight_names = {
        "ppo": "best_policy.pth",
        "rainbow": "best_rainbow.pth",
        "sac": "best_sac.pth",
        "sac_v2": "best_sac_v2.pth",
        "dqn": "best_dqn.pth",
        "qrdqn": "best_qrdqn.pth",
    }
    if args.path is None:
        args.path = str(ROOT / "rl_training" / "weights" / _weight_names[args.algo])

    gif_name = str(ROOT / "rl_training" / "replays" / f"best_{args.algo}.gif")

    policy = load_policy(args.path, args.algo)
    run_episode(policy, args.algo, render=True,
                save_gif=not args.no_gif, gif_path=gif_name)
