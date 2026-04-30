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
│   ├── afterstate_qrdqn.py     # Stage 4 Branch B — quantile V over afterstates (51 quantiles)
│   ├── afterstate_cnn.py       # Stage 4 Branch A — CNN over raw 20×10 board, scalar V
│   ├── afterstate_spr.py       # Stage 4 Branch D — SPR auxiliary loss on raw board CNN
│   ├── mcts.py                 # Stage 5a — expectimax planning over afterstate enum (depth=1,2+)
│   ├── chain_runs.py           # auto-chain: waits for PID, then runs queued training
│   ├── tune.py                 # generic MC hyperparameter search
│   ├── configs/
│   │   ├── sac_v2_default.json # default SAC v2 hyperparams
│   │   ├── sac_v2_plastic.json # SAC v2 + spectral/layer norm + RR=4
│   │   ├── sac_v2_search.json  # MC search space for sac_v2
│   │   ├── dqn_default.json    # default DQN hyperparams
│   │   ├── dqn_search.json     # MC search space for dqn
│   │   ├── qrdqn_default.json  # default QR-DQN hyperparams
│   │   ├── iqn_default.json    # default IQN hyperparams
│   │   ├── afterstate_dqn.json # afterstate hyperparams (32×32, eps decay over 1500 ep)
│   │   ├── afterstate_qrdqn.json # afterstate QR-DQN hyperparams (n_quantiles=51, kappa=1.0)
│   │   ├── afterstate_cnn.json # afterstate CNN hyperparams (conv [16,32], fc [64])
│   │   ├── afterstate_cnn_big.json # tuned CNN hyperparams (conv [32,64], fc [128,64], 6000ep)
│   │   └── afterstate_spr.json # Branch D SPR hyperparams (latent 128, spr_coef 0.5)
│   ├── weights/{algo}/YYYYMMDD_HHMMSS/  # per-algo, per-run dirs
│   │   ├── best.pth            # best weights for that run
│   │   ├── config.json         # hyperparams that generated it
│   │   └── ep{N:04d}.pth       # periodic checkpoints
│   ├── logs/{algo}/YYYYMMDD_HHMMSS/     # TensorBoard + config.json
│   └── replays/                # saved GIFs (best_{algo}.gif)
├── tetris/env/afterstate.py    # placement enumeration + 4-feature + raw-board state for afterstate agents
├── tests/                      # 147 tests (unit + integration)
├── train.py                    # orchestrator: --algo ppo/rainbow/.../afterstate/afterstate_qrdqn/afterstate_cnn
├── eval.py                     # unified eval: render+GIF (1 ep) or benchmark stats (N eps), all algos
├── main.py                     # human-playable
└── pyproject.toml
```

## Key interfaces
- **Env**: `Discrete(40)` = rotation * 10 + col. Dict obs: `{"obs": float32(78,), "mask": int8(40,)}`.
- **Rewards**: line clears (+1/3/5/8), game over (-10), survival (+0.05), height (-0.01/row), holes (-0.03/hole).
- **SAC wrappers**: `MaskedSACActor`/`ObsExtractCritic` — SAC doesn't handle dict obs natively (unlike DQN-family).
- **PlasticMLP** (sac_v2.py): drop-in Net replacement with spectral norm + layer norm, toggled via config booleans.
- **Afterstate path** (`tetris/env/afterstate.py` + `rl_training/afterstate_dqn.py`): standalone — bypasses TetrisEnv. Action key = `(rotation, x)` tuple, x in [-3, COLS-1] for full coverage. State = 4 hand-crafted features `[lines, holes, bumpiness, height]`. Reward = `1 + lines²·width − 2 if gameover` (nuno-faria formula). Network = 32×32 MLP with single scalar V(afterstate) output. Action selection enumerates all valid placements, picks max V.
- **eval.py**: path-aware builders for `sac_v2`, `qrdqn`, `iqn` — reads `config.json` from weights dir to auto-detect architecture.
- **eval.py benchmark mode** (`--episodes N`): standardized 30-ep greedy benchmark for all algos. Reports mean ± std for reward, **lines cleared**, **placements** (the latter two are reward-formula-independent — required for fair cross-algo comparison).

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
| 15 | **Afterstate QR-DQN (Stage 4 Branch B)** | **+1724 ± 899 reward, 99 ± 52 lines, 285 ± 130 placements** (30-ep greedy eval) | 3000ep, 51 quantiles, kappa=1.0, otherwise identical config to Run 14. +31% lines, +25% placements over scalar baseline at zero tuning. Distributional V helps even on the working paradigm. |
| 16 | Afterstate CNN (Stage 4 Branch A, small) | +85 ± 55 reward, **3.8 ± 3.0 lines, 37 ± 8 placements** (30-ep greedy eval) | 3000ep, conv [16,32] + fc [64], raw 20×10 board. **~20× fewer lines than 4-feature baseline.** Answer to "paradigm vs features?" — at this budget, the 4 hand-crafted features are doing major load-bearing work. Sample efficiency bottleneck confirmed. |
| 17 | Afterstate CNN (Stage 4 Branch A, big) | +356 ± 102 reward, **24.6 ± 7.3 lines, 100 ± 19 placements** (30-ep greedy eval) | 6000ep, conv [32,64] + fc [128,64], 2× updates/ep. **6× uplift over small CNN** — capacity and sample budget matter. Still **3× below 4-feature baseline** (76 lines/ep). Gap narrowing with scale but not closing → confirms SPR/BBF auxiliary losses (Branch D) are the right next step. |
| 18 | Afterstate SPR (Stage 4 Branch D) | +450 ± 173 reward, **31.3 ± 12.7 lines, 117 ± 32 placements** (30-ep greedy eval) | 6000ep, same arch as Branch A big + SPR aux loss (latent 128, action_emb 8, spr_coef 0.5). **+27% lines over Branch A big** (24.6 → 31.3). SPR helps but the gap to 4-feature baseline (76 lines) persists — still 2.4× below. Improvement is real but modest: SPR improves sample efficiency without closing the feature-engineering gap. |
| 19 | MuZero-lite afterstate (Stage 5b, original — buggy) | +32.7 ± 15.0 reward, **0.87 ± 1.20 lines, 26.1 ± 4.0 placements** (30-ep greedy eval) | 6000ep, CNN repr + learned dynamics + QR value head, k=5 unroll. **Complete failure** — 6 bugs identified post-run. See MuZero fix log below. |
| 20 | MuZero Fix 1: afterstate board in replay | **2.70 ± 1.55 lines, 35.4 ± 6.1 placements** (30-ep greedy eval) | 3000ep, 66k transitions. Fixing pre-action vs afterstate board in replay alone gives **3× uplift** (0.87→2.70 lines). Still QR value head (not yet fixed). Best eval during training: 75.8 reward at ep 2000. |
| 21 | MuZero Fix 2: scalar value head (MSE) | **1.17 ± 1.39 lines, 33.2 ± 4.0 placements** (30-ep greedy eval) | 3000ep, 62.5k transitions. Replacing QR head with Linear(1)+MSE actually regressed vs Fix 1 (2.70→1.17 lines). Best eval during training: 40.6 reward at ep 1450. Scalar head removed the broadcast bug but also removed the distributional variance signal that was helping exploration. |

## MuZero fix log (Stage 5b debugging)

Run 19 failed completely (0.87 lines/ep). Six bugs identified and being fixed sequentially. Each fix is a 3000ep run with 30-ep greedy eval.

| Fix | Description | Lines/ep (30-ep eval) | Transitions | Notes |
|-----|-------------|----------------------:|------------:|-------|
| Baseline (Run 19) | No fixes, 6000ep | 0.87 ± 1.20 | ~78k | QR head + pre-action board + tiny buffer |
| **Fix 1** | Store afterstate board in replay (not pre-action zeros) | **2.70 ± 1.55** | 66k | 3× improvement. Board bug was the single biggest issue. |
| Fix 2 | Scalar MSE value head (remove degenerate QR broadcast) | **1.17 ± 1.39** | 62.5k | Regression vs Fix 1 (2.70→1.17). Scalar MSE removed broadcast bug but also killed distributional variance signal. Fix 1+2 stacked. |
| Fix 3 | Scale replay to 50k transitions (was 500 episodes ≈ 7.5k) | TBD | — | Run b635bf2ux in progress. Also raises min_replay_size 50→5000 to prevent premature overfitting. Replay converted to transition-level capacity with FIFO episode eviction. |
| Fix 4 | Soft EMA target net (τ=0.005 Polyak) replaces hard copy every episode | TBD | — | Implemented; awaiting Fix 3 result before running. config: ema_tau=0.005. |
| Fix 5 | Separate hidden layer for reward head (dyn_r_hidden, latent//2) | TBD | — | Implemented; avoids conflicting gradients between z_next and r_hat paths. |
| Fix 6 | Gradient clipping (clip_grad_norm_=1.0) | TBD | — | Implemented; config: grad_clip=1.0. Gate is grad_clip>0 so disabled by default. |
| Fix 3 | Scale replay to 50k transitions | TBD | — | Pending |
| Fix 4 | Soft EMA target net (τ=0.005) | TBD | — | Pending |
| Fix 5 | Separate hidden layer for reward head | TBD | — | Pending |
| Fix 6 | Gradient clipping (norm=1.0) | TBD | — | Pending |

## Cross-algo benchmark (30-ep greedy eval via `eval_metrics.py`)

| Agent | Lines/ep (mean ± std) | Placements/ep (mean ± std) |
|-------|----------------------:|---------------------------:|
| DQN + PER | 0.97 ± 0.75 | 36.1 ± 3.0 |
| SAC v2 (best run, 0407_194747) | 13.30 ± 4.99 | 71.4 ± 13.4 |
| Afterstate DQN (extended) | 76.10 ± 51.12 | 227.4 ± 128.0 |
| **Afterstate QR-DQN (Branch B)** | **99.43 ± 52.08** | **285.2 ± 130.0** |
| Afterstate CNN (Branch A small, 3000ep) | 3.83 ± 2.97 | 36.97 ± 7.56 |
| Afterstate CNN (Branch A big, 6000ep)   | 24.60 ± 7.28 | 99.77 ± 18.50 |
| Afterstate SPR (Branch D, 6000ep)       | 31.33 ± 12.66 | 116.80 ± 31.86 |
| MCTS depth=1 (on Afterstate QR-DQN)    | 112 ± 88      | TBD            |
| MCTS depth=2 (on Afterstate QR-DQN)    | TBD (still running) | TBD      |
| MuZero-lite afterstate (Stage 5b)      | 0.87 ± 1.20   | 26.1 ± 4.0     |

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
15. **MuZero-style learned dynamics need an accurate world model to be useful** — k-step unrolled training on raw 20×10 board at 6000ep produced 0.87 lines/ep (vs 31 for plain SPR, 76 for 4-feature baseline). Learned dynamics compound the raw-board sample efficiency problem: the model must learn both board representation and accurate forward prediction simultaneously. At this budget, noise dominates signal. Applies whenever the dynamics are hard to learn from sparse rewards.
14. **Hand-crafted features are doing major work in the afterstate paradigm** — Branch A (raw 20×10 board → CNN → V) at 3000ep clears 3.8 lines/ep vs 76 for the 4-feature MLP, ~20× regression. The features `[lines, holes, bumpiness, height]` directly encode V-relevant quantities; learning them from raw boards under sparse nuno reward is sample-starved at this budget. Sample efficiency, not architecture, is the likely bottleneck. Empirically validates the A→D dependency: SPR/BBF-style auxiliary representation losses become well-motivated when raw input is the bottleneck.

## Roadmap

### Stage 2 — PPO baseline (done)
### Stage 3 — Distributional RL + plasticity (done)
- Rainbow DQN, SAC v1/v2, QR-DQN, IQN, spectral/layer norm, PER
- Conclusion: Q(s,a) family hit a wall on Tetris — plasticity loss + sparse rewards.

### Stage 3.5 — Afterstate paradigm shift (done — current SOTA agent)
- Afterstate DQN (nuno-faria style) with full action coverage: **76 ± 51 lines/ep**.
- Now the reference baseline for all future work.

### Stage 4 — Beating the afterstate baseline (non-planning)

Order matters here — they're not fully orthogonal:
1. **B (done)** — Distributional V over afterstates (QR over 4-feature input). +31% lines over scalar baseline.
2. **A (done)** — Raw board CNN. Small (3000ep): 3.8 lines/ep. Big (6000ep): 24.6 lines/ep. Gap vs 4-feature baseline (76 lines) narrowing with scale but not closing → confirms SPR needed.
3. **D (done)** — SPR auxiliary loss on raw board CNN. Transition model predicts next latent given current latent + (rot,dx) action embedding. Loss = TD(MSE) + 0.5 × SPR(cosine). 6000ep: 31.3 lines/ep (+27% over Branch A big). SPR helps but still 2.4× below 4-feature baseline. Conclusion: feature engineering gap is not bridgeable by auxiliary losses alone at this budget — planning (Stage 5) is the right next lever.

### Stage 5 — Planning
- **5a (partial)** — MCTS on top of afterstate enum, real simulator via `copy.deepcopy(game)`. V-net (Afterstate QR-DQN) as leaf evaluator. depth=1: **112 ± 88 lines/ep** (+13% over no-planning QR-DQN baseline of 99 lines). depth=2: in progress (slow — ~1470 evals/move).
- **5b (done — failed)** — MuZero-lite: CNN repr + learned dynamics (k=5 unroll) + QR value head on raw 20×10 board. 6000ep: **0.87 lines/ep**. Complete failure — learned dynamics on raw board is even more sample-starved than plain CNN V; adds noise not signal. Confirms MuZero-style planning requires an accurate learned model, which 6000ep can't provide for raw Tetris boards.

### Stage 6 — Engine extension + modern Tetris features
- Hold piece, T-spin detection, combo/B2B counters, perfect-clear bonus, garbage queue + cancel mechanics.
- Retrain afterstate baseline on the extended engine to confirm no regression.
- Prereq for Stage 7 — TETR.IO's attack table is meaningless without these features.

### Stage 7 — Versus mode + self-play
- 2-agent combat: line clears send garbage to opponent's board, normal Tetris top-out loses. Mechanics: https://tetris.wiki/TETR.IO#Mechanics
- ELO eval via league play + rule-based anchor opponents (greedy clearer, T-spin bot).
- Self-play infra: opponent pool / fictitious self-play / PSRO — naive agent-vs-current-self tends to find exploits and forget older opponents.
- Afterstate state must extend to include opponent context (incoming garbage, their combo/B2B, attack-readiness).

> Detailed notes for Stages 6–7 in `.claude/stage_6_7.md`. Revisit after Stages 4 and 5.

## Active runs
- MCTS depth=2 benchmark (30-ep, on Afterstate QR-DQN) — still running (task byu45o9n1).

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