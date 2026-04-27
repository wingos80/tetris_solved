# Tetris RL Project

## Project goal
Train an RL agent to play Tetris, progressing from basic PPO to SOTA techniques (distributional RL, plasticity fixes, planning).

## Project structure
```
tetris/                         # project root
├── tetris/                     # main package
│   ├── game.py                 # core logic, pygame-free
│   ├── renderer.py             # pygame rendering
│   └── env/
│       ├── tetris_env.py       # Gymnasium env, Discrete(40), dict obs
│       └── rewards.py          # reward shaping coefficients
├── rl_training/                # RL algorithm implementations + artifacts
│   ├── ppo.py                  # masked PPO (2×256)
│   ├── rainbow.py              # Rainbow DQN (C51+dueling+NoisyNet+PER)
│   ├── sac.py                  # Discrete SAC v1 (2×256, hardcoded)
│   ├── sac_v2.py               # SAC v2 (config-driven, PlasticMLP support)
│   ├── dqn.py                  # Double DQN (config-driven, 3×256)
│   ├── qrdqn.py                # QR-DQN (config-driven, 3×256)
│   ├── iqn.py                  # IQN (cosine embeddings, ObsExtractNet)
│   ├── afterstate_dqn.py       # afterstate value learning (nuno-faria style, 32×32)
│   ├── configs/
│   │   ├── sac_v2_default.json # default SAC v2 hyperparams
│   │   ├── sac_v2_plastic.json # SAC v2 + spectral/layer norm + RR=4
│   │   ├── sac_v2_search.json  # MC search space for sac_v2
│   │   ├── dqn_default.json    # default DQN hyperparams
│   │   ├── dqn_search.json     # MC search space for dqn
│   │   ├── qrdqn_default.json  # default QR-DQN hyperparams
│   │   ├── iqn_default.json    # default IQN hyperparams
│   │   └── afterstate_dqn.json # afterstate hyperparams (32×32, eps decay over 1500 ep)
│   ├── weights/{algo}/YYYYMMDD_HHMMSS/  # per-algo, per-run dirs
│   │   ├── best.pth            # best weights for that run
│   │   ├── config.json         # hyperparams that generated it
│   │   └── ep{N:04d}.pth       # periodic checkpoints
│   ├── logs/{algo}/YYYYMMDD_HHMMSS/     # TensorBoard + config.json
│   └── replays/                # saved GIFs (best_{algo}.gif)
├── tetris/env/afterstate.py    # placement enumeration + 4-feature state for afterstate agents
├── tests/                      # 147 tests (unit + integration)
├── train.py                    # orchestrator: --algo ppo/rainbow/sac/sac_v2/dqn/qrdqn/iqn/afterstate
├── tune.py                     # generic MC hyperparameter search
├── eval.py                     # render + GIF, auto-finds latest weights per algo
├── eval_metrics.py             # benchmark: 30-ep greedy eval reporting reward/lines/placements
├── chain_runs.py               # auto-chain: waits for PID, then runs queued training
├── main.py                     # human-playable
├── plot_training.py            # plots training curves from TensorBoard logs
└── pyproject.toml
```

## Key interfaces
- **Env**: `Discrete(40)` = rotation * 10 + col. Dict obs: `{"obs": float32(78,), "mask": int8(40,)}`.
- **Rewards**: line clears (+1/3/5/8), game over (-10), survival (+0.05), height (-0.01/row), holes (-0.03/hole).
- **SAC wrappers**: `MaskedSACActor`/`ObsExtractCritic` — SAC doesn't handle dict obs natively (unlike DQN-family).
- **PlasticMLP** (sac_v2.py): drop-in Net replacement with spectral norm + layer norm, toggled via config booleans.
- **Afterstate path** (`tetris/env/afterstate.py` + `rl_training/afterstate_dqn.py`): standalone — bypasses TetrisEnv. Action key = `(rotation, x)` tuple, x in [-3, COLS-1] for full coverage. State = 4 hand-crafted features `[lines, holes, bumpiness, height]`. Reward = `1 + lines²·width − 2 if gameover` (nuno-faria formula). Network = 32×32 MLP with single scalar V(afterstate) output. Action selection enumerates all valid placements, picks max V.
- **eval.py**: path-aware builders for `sac_v2`, `qrdqn`, `iqn` — reads `config.json` from weights dir to auto-detect architecture.
- **eval_metrics.py**: standardized 30-ep greedy benchmark for all algos. Reports mean ± std for reward, **lines cleared**, **placements** (the latter two are reward-formula-independent — required for fair cross-algo comparison).

## Python environment
- venv: `.\venv\Scripts\python.exe` (Python 3.11.5)
- torch 2.4.0+cu121, tianshou 0.5.1, gymnasium 1.2.3, numpy 1.26.4
- GPU: NVIDIA RTX 3050 Laptop, CUDA 12.6. **torch >= 2.5 crashes** (DLL issue) — pin 2.4.0.
- Tests: `.\venv\Scripts\python.exe -m pytest tests/ -v`

## Training runs

| Run | Algo | Best reward | Notes |
|-----|------|-------------|-------|
| 1-2 | PPO (no mask / old rewards) | -6.4 / -7 | Useless baselines |
| 3 | PPO masked, tuned | **-1.73 ± 3.48** | 200ep, PPO ceiling |
| 4 | Rainbow DQN | **+16.76 ± 11.22** | 400ep, big jump from PPO |
| 5 | SAC v1 | **+25.24 ± 16.60** | 400ep, best so far in shaped-reward regime |
| 6-7 | SAC v2 variants | +2.8 to +13.9 | Bug fixes, entropy tuning, RR experiments |
| 8 | QR-DQN (51q) | **-3.56** | 400ep, plasticity loss after ep ~80 |
| 9 | IQN | **-4.26** | 400ep, same plasticity regression pattern |
| 10 | SAC v2 plastic RR sweep | +2.8 to +13.9 | Specnorm/layernorm constrained learning, no improvement |
| 11 | SAC v2 + PER (best params) | **+2.59** | 400ep, much worse than no-PER (+13.9) — PER oversampled gameovers |
| 12 | DQN + PER | **-8.36** | 400ep, never learned — DQN-family plasticity issue |
| 13 | Afterstate DQN (x ∈ [0,9]) | +111.40 (nuno reward) | 3000ep, hit ceiling at ~6 lines/ep due to missing leftmost-wall placements |
| 14 | **Afterstate DQN (full x ∈ [-3,9])** | **+1437 ± 911 reward, 76 ± 51 lines, 227 ± 128 placements** (30-ep greedy eval) | 3000ep, ~60 min on RTX 3050. **Best agent by far.** Full 162-action coverage was the unlock. |

## Cross-algo benchmark (30-ep greedy eval via `eval_metrics.py`)

| Agent | Lines/ep (mean ± std) | Placements/ep (mean ± std) |
|-------|----------------------:|---------------------------:|
| DQN + PER | 0.97 ± 0.75 | 36.1 ± 3.0 |
| SAC v2 (best run, 0407_194747) | 13.30 ± 4.99 | 71.4 ± 13.4 |
| **Afterstate DQN (extended)** | **76.10 ± 51.12** | **227.4 ± 128.0** |

Lines and placements are the only fair comparison — reward formulas differ across agents (shaped rewards for SAC/DQN family, nuno formula for afterstate).

## Lessons learned
1. **Action masking is critical** — without it, agent wastes episodes on invalid moves.
2. **tianshou `train_fn` fires per collect step (~26×/epoch)** — dedup guard needed: `if epoch != _last[0]`.
3. **SAC alpha crash** — target entropy vs full 40-action space overshoots masked space (~15-25). Use `entropy_frac` < 1.
4. **RR > 1 needs plasticity fixes** — RR=3 underperformed RR=1 without spectral norm/resets.
5. **DQN-family has native masking**; SAC/PPO need custom wrappers.
6. **Rainbow policy must `.to(DEVICE)`** — C51 `self.support` stays on CPU otherwise.
7. **Alpha tuple bug** — `DiscreteSACPolicy(alpha=(target_entropy, log_alpha, optim))` — passing `cfg["alpha"]` (0.2) instead of computed `target_entropy` (-1.84) causes alpha to spiral to infinity. "Correct type, wrong semantic."
8. **PlasticMLP needs torch.as_tensor()** — unlike tianshou's Net, custom modules don't auto-convert numpy inputs.
9. **PER hurts SAC on Tetris** — oversampling rare game-over transitions makes the policy over-optimize for avoidance instead of general play. Importance-sampling β=0.4 didn't compensate.
10. **Reward numbers across agents are not comparable** — different shaping. Use lines/ep + placements/ep via `eval_metrics.py` for objective benchmarks.
11. **TetrisEnv `Discrete(40)` action space loses 12/162 valid placements** — block.x ∈ [0,9] excludes x=−1 to −3 placements where the bbox extends past the left edge but real cells stay on board. This caps any agent learning to "use the leftmost column". The afterstate path uses tuple keys `(rot, x)` with x ∈ [-3, 9] for full coverage.
12. **Afterstate value learning >> Q(s,a) on Tetris** — V(afterstate) over enumerated placements with hand-crafted state, 32×32 net, ~60 min training: 6× more lines/ep than our best SAC v2 (3×256, 400 epochs). Paradigm shift, not architectural improvement.
13. **Custom training modules must derive device from net params, not a global** — `next(net.parameters()).device`. Hardcoding `DEVICE` global breaks CPU tests / multi-GPU. Caught by a constant-target overfit test in afterstate_dqn.

## Roadmap

### Stage 2 — PPO baseline (done)
### Stage 3 — Distributional RL + plasticity (done)
- Rainbow DQN, SAC v1/v2, QR-DQN, IQN, spectral/layer norm, PER
- Conclusion: Q(s,a) family hit a wall on Tetris — plasticity loss + sparse rewards.

### Stage 3.5 — Afterstate paradigm shift (done — current SOTA agent)
- Afterstate DQN (nuno-faria style) with full action coverage: **76 ± 51 lines/ep**.
- Now the reference baseline for all future work.

### Stage 4 — Beating the afterstate baseline
Branch options:
- **A** — Richer afterstate features (raw board input + learned embedding instead of 4 hand-crafted features). Tests whether feature engineering or paradigm is doing the work.
- **B** — Distributional V over afterstates (QR over 4-feature input). Combines our distrib RL with the working paradigm.
- **C** — MCTS on top of afterstate enum. The enum is already 1-step planning; MCTS is the natural extension.
- **D** — SPR / BBF on afterstate inputs. Sample efficiency improvements.

### Stage 5 — Planning
- MCTS via `copy.deepcopy(game)`, Stochastic MuZero / EfficientZero V2

### Stage 6 — Competition
- PvP mode (send lines), timed challenge vs AI

## Active runs
None — all queued runs complete.

## Conventions
- Minimal SLOC — no unnecessary abstractions.
- Absolute imports (`from tetris.game import ...`).
- `tetris/game.py` must stay pygame-free.
- Standard Tetris scoring: [0, 100, 300, 500, 800] for [0, 1, 2, 3, 4] lines.
- All training runs save to `weights/{algo}/YYYYMMDD_HHMMSS/` with `best.pth` + `config.json`.

## Errors to be mindful of
- correct type, wrong semantic
  - passing default alpha value as target entropy in the `alpha` tuple to `DiscreteSACPolicy(..., alpha=)`
- device hardcoding via global (use `next(net.parameters()).device` in custom training loops)
- restricted action space silently capping agents — measure coverage before assuming the env is complete