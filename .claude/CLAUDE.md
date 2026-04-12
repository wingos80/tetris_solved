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
│   ├── configs/
│   │   ├── sac_v2_default.json # default SAC v2 hyperparams
│   │   ├── sac_v2_plastic.json # SAC v2 + spectral/layer norm + RR=4
│   │   ├── sac_v2_search.json  # MC search space for sac_v2
│   │   ├── dqn_default.json    # default DQN hyperparams
│   │   ├── dqn_search.json     # MC search space for dqn
│   │   ├── qrdqn_default.json  # default QR-DQN hyperparams
│   │   └── iqn_default.json    # default IQN hyperparams
│   ├── weights/{algo}/YYYYMMDD_HHMMSS/  # per-algo, per-run dirs
│   │   ├── best.pth            # best weights for that run
│   │   ├── config.json         # hyperparams that generated it
│   │   └── ep{N:04d}.pth       # periodic checkpoints
│   ├── logs/{algo}/YYYYMMDD_HHMMSS/     # TensorBoard + config.json
│   └── replays/                # saved GIFs (best_{algo}.gif)
├── tests/                      # 87 tests (unit + integration)
├── train.py                    # orchestrator: --algo ppo/rainbow/sac/sac_v2/dqn/qrdqn/iqn
├── tune.py                     # generic MC hyperparameter search
├── eval.py                     # render + GIF, auto-finds latest weights per algo
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
- **eval.py**: path-aware builders for `sac_v2`, `qrdqn`, `iqn` — reads `config.json` from weights dir to auto-detect architecture.

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
| 5 | SAC v1 | **+25.24 ± 16.60** | 400ep, best so far. No bumpiness reward. |
| 6-7 | SAC v2 variants | +2.8 to +13.9 | Bug fixes, entropy tuning, RR experiments |
| 8 | QR-DQN (51q) | **-3.56** | 400ep, plasticity loss after ep ~80 |
| 9 | IQN | **-4.26** | 400ep, same plasticity regression pattern |
| 10 | SAC v2 plastic RR sweep | training... | 3 runs: RR=0.3/1.0/4.0, specnorm+layernorm |

## Lessons learned
1. **Action masking is critical** — without it, agent wastes episodes on invalid moves.
2. **tianshou `train_fn` fires per collect step (~26×/epoch)** — dedup guard needed: `if epoch != _last[0]`.
3. **SAC alpha crash** — target entropy vs full 40-action space overshoots masked space (~15-25). Use `entropy_frac` < 1.
4. **RR > 1 needs plasticity fixes** — RR=3 underperformed RR=1 without spectral norm/resets.
5. **DQN-family has native masking**; SAC/PPO need custom wrappers.
6. **Rainbow policy must `.to(DEVICE)`** — C51 `self.support` stays on CPU otherwise.
7. **Alpha tuple bug** — `DiscreteSACPolicy(alpha=(target_entropy, log_alpha, optim))` — passing `cfg["alpha"]` (0.2) instead of computed `target_entropy` (-1.84) causes alpha to spiral to infinity. "Correct type, wrong semantic."
8. **PlasticMLP needs torch.as_tensor()** — unlike tianshou's Net, custom modules don't auto-convert numpy inputs.

## Roadmap

### Stage 2 — PPO baseline (done)
### Stage 3 — Distributional RL + plasticity (in progress)
- ~~Rainbow DQN~~, ~~SAC v1/v2~~, ~~QR-DQN~~, ~~IQN~~, ~~spectral/layer norm~~
- **Running**: SAC v2 plastic RR sweep (RR=0.3 → 1.0 → 4.0) via `chain_runs.py`
- QR-DQN and IQN done but underwhelming (plasticity loss)
- **Remaining**: analyze RR sweep results, decide next algo direction

### Stage 4 — Representation & sample efficiency
- SPR (self-predictive), BBF (scaled nets + resets + SPR), hold actions (Discrete(80))

### Stage 5 — Planning
- MCTS via `copy.deepcopy(game)`, Stochastic MuZero / EfficientZero V2

### Stage 6 — Competition
- PvP mode (send lines), timed challenge vs AI

## Active runs
| Run | Config | Status |
|-----|--------|--------|
| SAC v2 plastic RR=0.3 | specnorm+layernorm, 400ep | **Running** (chain_runs.py) |
| SAC v2 plastic RR=1.0 | specnorm+layernorm, 400ep | Queued |
| SAC v2 plastic RR=4.0 | specnorm+layernorm+resets, 400ep | Queued |

## Conventions
- Minimal SLOC — no unnecessary abstractions.
- Absolute imports (`from tetris.game import ...`).
- `tetris/game.py` must stay pygame-free.
- Standard Tetris scoring: [0, 100, 300, 500, 800] for [0, 1, 2, 3, 4] lines.
- All training runs save to `weights/{algo}/YYYYMMDD_HHMMSS/` with `best.pth` + `config.json`.

## Errors to be mindful of
- correct type, wrong semantic
  - passing default alpha value as target entropy in the `alpha` tuple to `DiscreteSACPolicy(..., alpha=)`