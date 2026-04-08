# Tetris RL Project

## Project goal
Train an RL agent to play Tetris, progressing from basic PPO to SOTA techniques (distributional RL like in https://arxiv.org/abs/1707.06887 MCTS/MuZero-style planning).

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
│   ├── ppo.py                  # masked PPO
│   ├── rainbow.py              # Rainbow DQN (C51+dueling+NoisyNet+PER)
│   ├── sac.py                  # Discrete SAC v1 (auto-alpha, 2×256 nets)
│   ├── sac_v2.py               # Discrete SAC v2 (config-driven, 3×256 nets, run_trial)
│   ├── dqn.py                  # Double DQN (config-driven, 3×256 nets, run_trial)
│   ├── configs/
│   │   ├── sac_v2_default.json # default hyperparams for sac_v2
│   │   ├── sac_v2_search.json  # MC search space for sac_v2
│   │   ├── dqn_default.json    # default hyperparams for dqn
│   │   └── dqn_search.json     # MC search space for dqn
│   ├── weights/
│   │   ├── best_policy.pth         # PPO
│   │   ├── best_rainbow.pth        # Rainbow
│   │   ├── best_sac.pth            # SAC v1
│   │   └── YYYYMMDD_HHMMSS/        # sac_v2/dqn: one dir per run
│   │       ├── best.pth            # best weights for that run
│   │       ├── config.json         # hyperparams that generated it
│   │       └── ep{N:04d}.pth       # periodic checkpoints
│   ├── logs/                   # TensorBoard logs + tune results
│   │   ├── sac_v2/
│   │   │   └── YYYYMMDD_HHMMSS/    # one subdir per run
│   │   │       └── config.json     # hyperparams for that run
│   │   ├── dqn/
│   │   ├── {algo}_tune_results.json   # all MC trial scores
│   │   └── {algo}_tune_winner.json    # best config, ready for --config
│   └── replays/                # saved GIFs (best_{algo}.gif)
├── tests/
│   ├── unit/                   # test_block, test_game, test_rewards (49 tests)
│   └── integration/            # test_env, test_training (26 tests)
├── train.py                    # root orchestrator: --algo ppo/rainbow/sac/sac_v2/dqn/all
├── tune.py                     # generic MC hyperparameter search: --algo sac_v2/dqn
├── eval.py                     # render + GIF: --algo ppo/rainbow/sac/sac_v2/dqn
├── main.py                     # human-playable
├── plot_training.py            # plots training curves from TensorBoard logs
├── watch_and_eval.py           # watches training PIDs, auto-runs eval when done
├── run3_log.txt                # PPO run 3 training log
└── pyproject.toml
```

### Key files
- `tetris/env/tetris_env.py` — `Discrete(40)` = rotation * 10 + col. Dict obs: `{"obs": float32(78,), "mask": int8(40,)}`.
- `tetris/env/rewards.py` — line clears (+1/3/5/8), game over (-10), survival (+0.05), height (-0.01/row), holes (-0.03/hole).
- `rl_training/ppo.py` — tianshou PPO. `MaskedActor`/`ObsExtractCritic` wrappers (256×256 nets).
- `rl_training/rainbow.py` — `RainbowNet` (dueling+NoisyLinear+C51 atoms), `PrioritizedVectorReplayBuffer`. Policy must `.to(DEVICE)` to move C51's `self.support` to GPU.
- `rl_training/sac.py` — `MaskedSACActor`/`ObsExtractCritic` wrappers (SAC doesn't handle dict obs natively unlike DQN-family). 2×256 nets, hardcoded hyperparams.
- `rl_training/sac_v2.py` — config-driven SAC. `load_config(path)`, `run_trial(cfg, epochs)`, `main(max_epoch, config_path)`. 3×256 default.
- `rl_training/dqn.py` — Double DQN. `QNet` MLP handles `obs.obs` extraction. `load_config`, `run_trial`, `main`. 3×256 default.
- `tune.py` — generic MC search. Reads `{algo}_default.json` + `{algo}_search.json`, calls `module.run_trial()`. Writes winner config ready for `--config`.
- `eval.py` — Multi-algo evaluator. `--algo ppo/rainbow/sac/sac_v2/dqn`. GIFs → `rl_training/replays/best_{algo}.gif`. Uses `sys.path.insert(0, "rl_training/")` to import model classes.
- `train.py` — Orchestrator: selects algo, passes `--epochs` and `--config` overrides, calls `rl_training/*.main()`.

### Tests (75 total)
Run: `.\venv\Scripts\python.exe -m pytest tests/ -v`

## Python environment
- Use the venv at `./venv/` (Python 3.11.5 from Anaconda).
- Run via: `.\venv\Scripts\python.exe <script>`
- torch 2.4.0+cu121, tianshou 0.5.1, gymnasium 1.2.3, numpy 1.26.4, Pillow 12.1.1
- GPU: NVIDIA RTX 3050 Laptop GPU, CUDA 12.6 drivers, torch cu121 build.
- torch >= 2.5 has DLL issues on this Windows machine — stick with 2.4.0.

## Training runs

| Run | Algo | Epochs | Best reward | Ep len | Notes |
|-----|------|--------|-------------|--------|-------|
| 1 | PPO no masking | 200 | -6.44 (ep143) | ~12 | useless — invalid actions dominate |
| 2 | PPO masked, old rewards | 200 | ~-7 | ~12 | agent learned to die fast (game_over -5 < survival cost) |
| 3 | PPO masked, tuned rewards | 200 | **-1.73 ± 3.48** (ep156) | ~48 | 2M steps, ~13.5h, high variance, PPO ceiling |
| 4 | Rainbow DQN | 400 | **+16.76 ± 11.22** (ep325) | — | 4M steps, ~25.7h, massive jump from PPO. High variance. |
| 5 | Discrete SAC v1 | 400 | **+25.24 ± 16.60** (ep382) | — | 4M steps, ~25.8h, best so far. High variance. |
| 6 | SAC v2 (pre-fix) | ~180 | **+2.837** (~900k steps) | — | Multi-fire reset bug poisoned all resets; alpha crashed to 0.05; best weights lost (overwritten). |
| 7A | SAC v2 (RR=3, resets) | 200 | TBD | — | entropy_frac=0.3; multi-fire bug (poisoned); RR=3 underperformed |
| 7B | SAC v2 (RR=3, no resets) | 200 | TBD | — | entropy_frac=0.3; no reset bug here; RR=3 underperformed |
| 7C | SAC v2 (RR=1, resets) | 200 | TBD | — | entropy_frac=0.3; multi-fire bug; best of A/B/C |
| 7D | SAC v2 (RR=1, resets) | 400 | — | — | entropy_frac=0.5, bugs fixed; **currently running** |

Reward changes Run 2→3: game_over -5→-10, height -0.05→-0.01, holes -0.1→-0.03, added survival +0.05.

## Lessons learned
1. **Action masking is critical** — without it agent learns "avoid invalid" not Tetris. Doubles effective episode length.
2. **tianshou PPO has no native masking** — wrap `Actor` with `MaskedActor` (extracts `obs.obs`, applies `logits[~mask]=-1e8`). `Critic` needs `ObsExtractCritic` wrapper too.
3. **DQN-family has native masking** — `DQNPolicy.compute_q_value()` handles `obs.mask` automatically. No wrappers needed.
4. **Reward shaping can backfire** — heavy per-step penalties make dying cheaper than surviving. Game-over penalty must dominate.
5. **torch >= 2.5 DLL crash on Windows** — pin to 2.4.0 (c10.dll VC++ mismatch). numpy < 2 required.
6. **Rainbow: policy must `.to(DEVICE)`** — C51Policy holds `self.support` as `nn.Parameter`; it stays on CPU unless the policy itself is moved.
7. **tianshou `train_fn` fires per collect step, not per epoch** — ~26× per epoch. Any stateful per-epoch logic (resets, LR schedules) needs a dedup guard: `if epoch != _last_epoch[0]: _last_epoch[0] = epoch; ...`.
8. **SAC alpha crash with masked actions** — `log_alpha` inits at 0 → α=1.0 regardless of config. Target entropy vs full action space (40) overshoots masked valid space (~15-25) → α driven to near-zero. Use `entropy_frac=0.3` or dynamic target `= log(avg_valid_actions)`.
9. **Replay ratio 1 > 3 on this env** — more gradient steps per sample hurt; likely unstable Q-targets with small buffer and fast policy change.

## Roadmap

### Stage 2 — PPO baseline (complete)
~~Runs 1-3: masking, reward tuning, best -1.73~~

### Stage 3 — Distributional RL (in progress)
4. ~~**Rainbow DQN**~~ — running (400 epochs). C51+double+PER+dueling+NoisyNet. tianshou `RainbowPolicy`.
5. ~~**Discrete SAC**~~ — running (400 epochs). Auto-tuned alpha. tianshou `DiscreteSACPolicy`.
6. **QR-DQN** — simpler distributional baseline, no V_min/V_max. tianshou `QRDQNPolicy`.
7. **Spectral norm + layer norm** — combats plasticity loss. One specnorm layer ≈ Rainbow-level (arxiv 2105.05246).
8. **Higher replay ratio** (RR 4-8) + periodic layer resets for plasticity.
9. **IQN** — risk-sensitive play, CVaR action selection. tianshou `IQNPolicy`.

### Stage 4 — Representation & sample efficiency
10. **SPR** — self-predictive auxiliary loss. +55% Atari 100K (arxiv 2007.05929).
11. **BBF** — scaled nets + resets + increasing gamma + SPR. Superhuman Atari 100K (arxiv 2305.19452).
12. **Hold actions** — Discrete(80), actions 40-79 = hold+place.

### Stage 5 — Planning
13. **MCTS** — known model via `copy.deepcopy(game)`. See github.com/hrpan/tetris_mcts.
14. **Stochastic MuZero / EfficientZero V2** — handles stochastic next-piece (arxiv 2403.00564).

### Stage 6 — Competition
15. Add in competition mode vs user and see who can clear more lines in 1 minute
16. Add mode seen on jstris or tetr.io, where you duel players and send lines to each other based on how many lines you compeleted.

## Research notes

### Distributional RL
Learn Z(s,a) distribution instead of scalar Q. ~2x Atari improvement, risk-sensitive policies. All in tianshou 0.5.1:
- **C51**: fixed atoms, learned probs. Needs V_min/V_max. Used in Rainbow.
- **QR-DQN**: fixed probs, learned quantile locs. No V_min/V_max.
- **IQN**: continuous quantile fn via tau embeddings. CVaR-capable.
- **Rainbow**: C51 + double DQN + PER + dueling + n-step + NoisyNet.

V_min/V_max for this env: -20/25 (worst = immediate death -10, best = sustained line clears ~+20-30 discounted).

### Plasticity & sample efficiency
- **High replay ratio** (RR 4-8): more gradient steps per sample. Needs plasticity mitigation.
- **Plasticity fixes**: periodic final-layer resets (BBF), spectral norm, layer norm + residuals.
- **TQC**: truncate upper quantiles in QR-DQN to combat Q overestimation.
- **SPR**: predict future latent states as auxiliary loss. Foundation of BBF.

### Tetris-specific
- Stochastic (random next piece) → need Stochastic MuZero for planning.
- Known transition model → `copy.deepcopy(game)` = perfect MCTS simulator.
- Small action space (40, effective ~15-25 with masking) → entropy exploration (SAC) less valuable.

## Useful paths
- **Memory files**: `C:\Users\micky\.claude\projects\c--Users-micky-Desktop-wings-stuff-coding-my-stuff-random-projs-tetris\memory\`
- **venv python**: `.\venv\Scripts\python.exe`

## Development history (condensed)

### 2026-03-31: Rainbow + SAC v1
Built `rainbow.py` (RainbowNet: dueling+NoisyLinear+C51+PER) and `sac.py` (MaskedSACActor+ObsExtractCritic wrappers). `eval.py` + `train.py` orchestrator. Bugs: C51 `self.support` stays on CPU unless `policy.to(DEVICE)` called; SAC n-step returns require trainer's `process_fn`, not `policy.learn()` directly.

### 2026-04-01: Tuning infrastructure + DQN
Built `sac_v2.py` (config-driven, 3×256, `run_trial()`), `dqn.py` (Double DQN, same interface), `tune.py` (generic MC search), JSON configs, updated `eval.py` + `train.py`.

### 2026-04-07: Bug fixes + comparative runs
Diagnosed and fixed two critical sac_v2 bugs:
1. **Multi-fire reset**: tianshou calls `train_fn` ~26× per epoch; actor head was being reset every collect step. Fixed with `_last_reset_epoch = [-1]` guard in both `main()` and `run_trial()`.
2. **Alpha crash to ~0.05**: `log_alpha` inits at 0 → α=1.0; target entropy uses full 40-action space but masking limits valid to ~15-25 → target unachievable → α collapses. Fixed short-term with `entropy_frac=0.3`. Long-term: dynamic target `= log(avg_valid_actions)` documented in `.claude/plasticity_fixes.md`.

Infrastructure: timestamped run dirs for TensorBoard (`logs/sac_v2/YYYYMMDD_HHMMSS/`) and weights (`weights/YYYYMMDD_HHMMSS/`); `config.json` saved alongside both.

Comparative runs A/B/C (200ep, entropy_frac=0.3): RR=1 outperformed RR=3; resets marginal. All runs had multi-fire bug. Run D (400ep, entropy_frac=0.5, RR=1, bug-fixed) launched — clean analog of best pre-fix run (+2.837 peak).

## Active runs
| Task | Config | Status |
|------|--------|--------|
| `b1wk46s3x` | sac_run_d.json (entropy_frac=0.5, RR=1, resets, 400ep) | **Running** |

## Conventions
- Minimal SLOC preferred — no unnecessary abstractions.
- Absolute imports everywhere (`from tetris.game import ...`, not relative).
- `tetris/game.py` must stay pygame-free (critical for headless RL training).
- Standard Tetris scoring: [0, 100, 300, 500, 800] for [0, 1, 2, 3, 4] lines.
