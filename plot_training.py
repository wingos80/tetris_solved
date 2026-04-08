"""Plot test_reward over epoch from current Rainbow + SAC training output files."""

import re
import sys
import matplotlib.pyplot as plt

LOGS = {
    "rainbow": (
        r"C:\Users\micky\AppData\Local\Temp\claude"
        r"\c--Users-micky-Desktop-wings-stuff-coding-my-stuff-random-projs-tetris"
        r"\4c845da6-cdaf-4caa-8d97-df5d24876e88\tasks\bdforbvzi.output"
    ),
    "sac": (
        r"C:\Users\micky\AppData\Local\Temp\claude"
        r"\c--Users-micky-Desktop-wings-stuff-coding-my-stuff-random-projs-tetris"
        r"\4c845da6-cdaf-4caa-8d97-df5d24876e88\tasks\bx0rct825.output"
    ),
}

# Epoch #104: test_reward: 2.813000 ± 9.293086, best_reward: ...
PATTERN = re.compile(
    r"Epoch #(\d+): test_reward: ([-\d.]+) [^\d]+ ([\d.]+)"
)


def parse(path):
    epochs, rewards, stds = [], [], []
    try:
        with open(path, "r", errors="replace") as f:
            for line in f:
                m = PATTERN.search(line)
                if m:
                    epochs.append(int(m.group(1)))
                    rewards.append(float(m.group(2)))
                    stds.append(float(m.group(3)))
    except FileNotFoundError:
        print(f"  [warn] not found: {path}", file=sys.stderr)
    return epochs, rewards, stds


fig, ax = plt.subplots(figsize=(10, 5))
ax.axhline(-1.73, color="gray", linestyle="--", linewidth=1, label="PPO best (−1.73)")

for algo, path in LOGS.items():
    epochs, rewards, stds = parse(path)
    if not epochs:
        print(f"  [warn] no data parsed for {algo}")
        continue
    rewards_arr = rewards
    stds_arr = stds
    ax.plot(epochs, rewards_arr, label=algo)
    lo = [r - s for r, s in zip(rewards_arr, stds_arr)]
    hi = [r + s for r, s in zip(rewards_arr, stds_arr)]
    ax.fill_between(epochs, lo, hi, alpha=0.15)
    print(f"{algo}: {len(epochs)} epochs, latest rew={rewards[-1]:.2f} ± {stds[-1]:.2f} (ep {epochs[-1]})")

ax.set_xlabel("Epoch")
ax.set_ylabel("Test reward (mean ± std)")
ax.set_title("Training progress")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("rl_training/logs/training_progress.png", dpi=150)
print("Saved rl_training/logs/training_progress.png")
plt.show()
