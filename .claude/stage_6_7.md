# Stages 6 & 7 — Notes (revisit after Stages 4 & 5)

Original Stage 6 idea: 2-agent TETR.IO-style combat (clear lines → send garbage to opponent). On reflection this splits cleanly into two stages because the engine prerequisites are roughly as much work as the agent training.

## Stage 6 — Engine extension

`tetris/game.py` currently models vanilla Tetris: 7 pieces, line clears, gravity, top-out. TETR.IO's attack table relies on features the engine doesn't have. Without them, "combat" reduces to "whoever clears lines first wins" — closer to NES Tetris versus, not TETR.IO.

Features needed (rough order of difficulty):
- **Hold piece** — small state addition (`hold_piece`, `hold_used_this_drop` flag), but afterstate enumeration roughly doubles (each placement now branches on hold/no-hold).
- **T-spin detection** — 3-corner rule, kick-table-aware. Affects scoring and B2B chain.
- **Combo counter** — increments on consecutive line-clearing pieces, resets on a non-clearing drop.
- **B2B (back-to-back) counter** — tracks consecutive "difficult" clears (tetris, T-spin). Affects attack table.
- **Perfect clear bonus** — board-empty detection after a clear.
- **Garbage queue + cancel window** — incoming garbage sits in a queue for N frames; outgoing attacks cancel it before it spawns. This is the part that makes versus *interactive* rather than two parallel games.

Retrain the afterstate baseline on the extended engine before Stage 7 — confirms no regression and gives a sane single-agent reference for the multi-agent stage.

Open questions:
- SRS kicks vs. simpler kick table? TETR.IO uses SRS+ with extra kicks for T-spins. Decide based on whether we want the agent to discover T-spin doubles/triples.
- 7-bag randomizer is already standard in modern Tetris — verify our piece generator already does this, swap in if not.

## Stage 7 — Versus mode + self-play

Three big subsystems beyond the algorithm:

### 1. Multi-agent training infrastructure
Naive self-play (agent vs. current weights) converges to weird equilibria — agent finds exploits against itself, forgets how to beat older versions, then loses to them again. Options:
- **Opponent pool / league play** (AlphaStar style) — keep snapshots, sample opponents from the pool.
- **Fictitious self-play** — train against the average of past policies.
- **PSRO** (policy-space response oracles) — heavier, principled, probably overkill for Tetris.

Start with a simple opponent pool (e.g. 10 most recent + a few rule-based anchors). Upgrade if it stagnates.

### 2. Eval via ELO
No fixed scoreboard in versus. Need:
- **Anchor opponents** — rule-based bots that don't change: greedy line-clearer, height-minimizer, T-spin bot, random-but-legal.
- **League** — round-robin or Swiss between current agent + snapshots + anchors.
- **ELO updates** after each match. Anchors give absolute reference; snapshots show training progress.

### 3. Afterstate paradigm extension
Right now V(afterstate) takes 4 features of *your* board. In versus, ideal placement is contextual:
- Stack tall for a tetris when free.
- Defend (low/flat board) when garbage is incoming.
- Hold for T-spin if opponent has weak attack-readiness.

State must include opponent context. Probably:
- Your 4 features + opponent's 4 features
- Incoming garbage queue (count + frames-until-spawn)
- Your combo/B2B/hold state, opponent's combo/B2B
- Maybe attack-readiness summary (next piece + hold piece)

Could test whether the afterstate paradigm still holds at this complexity, or whether a learned-model approach (Stage 5b MuZero) becomes more competitive in the multi-agent setting.

### Action space
With hold piece, the action becomes `(rot, x, use_hold)` — afterstate enumeration grows but stays tractable (~324 actions). T-spin tucks/spins might require more permissive action enumeration (rotation-after-touchdown), or be left for the network to discover via finishing-move kicks during the drop animation.

## Open design questions to resolve before starting

- Full TETR.IO attack table or simplified subset? Suggest: start with simplified (single/double/triple/tetris/T-spin send values, B2B +1, combo table from Tetris Guideline). Add cancel windows. Skip exotic moves (all-spin, etc.) initially.
- Reward shaping for versus — pure win/loss is sparse; might need shaping on (lines sent − lines received) or APM.
- Game length cap for training? Real versus has no cap; for stable training probably need a max-frames cutoff.
- Symmetric self-play (both agents same policy) or asymmetric (champion vs. challenger)?

## Why we deferred this

Stages 4 and 5 still have headroom on the single-agent problem. Branch A/B/D in Stage 4 will tell us whether the afterstate paradigm scales with feature complexity — directly relevant to whether it scales to versus state too. Stage 5 (MCTS, then MuZero) gives us a planning baseline. Coming back to versus with those answers in hand makes the design choices much cleaner.
