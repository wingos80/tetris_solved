# Plasticity & Catastrophic Forgetting Fixes

## Problem
SAC/Rainbow plateau around epoch 100-150 due to plasticity loss — dead neurons, saturated weights, near-zero gradients on ReLUs. Networks lose capacity to form new representations even with new experience.

Symptom: flat reward with high variance, no upward trend despite continued training.

## Fixes tried / considered

### 1. Periodic output head resets (implemented, too aggressive)
Reset final linear layer to random init every N epochs. Restores learning capacity in the head while keeping learned features in earlier layers.

**Status**: Implemented in `sac_v2.py` / `dqn.py` (`reset_freq=100`). Causes reward crash at exactly reset epochs (500k, 1M steps). All 3 heads (actor + 2 critics) reset simultaneously — too disruptive.

**Source**: BBF paper (arxiv 2305.19452), also used in DER, SR-SPR.

---

### 2. Actor-only resets (next to try)
Reset only `actor.last` every N epochs. Leave critics intact so Q-value estimates survive the reset — actor recovers quickly since it still gets good gradients from stable critics.

**Rationale**: Critic resets cause the policy collapse. The actor depends on critic gradients; wiping critics gives the actor garbage signal for many epochs. Actor plasticity loss is the real bottleneck anyway.

**How to apply**: In `sac_v2.py` `train_fn`, remove `critic1.last.reset_parameters()` and `critic2.last.reset_parameters()`.

---

### 3. Shrink-and-perturb (softer alternative to hard resets)
Instead of `reset_parameters()`, do `w = 0.9*w + 0.1*noise` on the output head. Moves weights toward zero while injecting exploration noise — much less disruptive than full reset.

**Rationale**: Preserves some learned structure while restoring plasticity. Avoids the sudden Q-value collapse from hard resets.

**Source**: "Plasticity in Deep Continual Learning" (Nikishin et al.), also used in BBF.

---

### 4. Staggered resets
Reset actor at epoch 100, critic1 at 133, critic2 at 166. Never all at once. Spreads the disruption so at least 2 of 3 networks are stable at any time.

**Rationale**: Reduces the simultaneous shock to the policy. Each network resets while the others provide stable gradients.

---

### 5. Gradient clipping
Cap gradient norm at 0.5-1.0 in actor/critic optimizers. Prevents sudden large updates from overwriting learned features.

**How to apply**: Pass `max_grad_norm` to tianshou policy, or manually clip before each optimizer step.

---

### 6. Lower learning rate
Drop `lr_actor`/`lr_critic` from 1e-4 to 3e-5. More conservative updates = slower forgetting between resets.

**Tradeoff**: Slower initial learning but more stable long-term.

---

### 7. Higher replay ratio (RR 4-8) — tried, failed
More gradient steps per env step. Theoretically increases sample efficiency but **accelerates plasticity loss on cold buffer** — caused complete training failure when buffer was <10% full.

**Status**: Reverted to RR=1. Should only be used paired with a warmup period (e.g. skip updates until buffer has 50k+ transitions) or with aggressive plasticity fixes already in place.

**Source**: SR-SPR (arxiv 2203.00893), BBF (arxiv 2305.19452).

---

### 8. Layer norm + spectral norm
Add LayerNorm after each hidden layer, spectral norm on one layer. Combats weight explosion and keeps gradient flow healthy.

**Source**: "Revisiting the Plasticity of Neural Networks for Deep RL" (arxiv 2105.05246). One specnorm layer ≈ Rainbow-level improvement on Atari.

---

---

### 9. Dynamic target entropy based on valid actions
Instead of `target_entropy = -log(1/40) * entropy_frac`, compute:
`target_entropy = log(avg_valid_actions)` where avg_valid_actions ≈ mean mask sum over a rollout.

**Rationale**: With masking, only ~15-25 of 40 actions are valid. The theoretical max entropy is `log(20) ≈ 3.0`, not `log(40) ≈ 3.69`. Setting target relative to full action space makes SAC over-aggressively drive α to zero. Dynamic target anchors entropy to what the policy can actually achieve.

**How to apply**: Collect mean valid action count during `train_fn` rollouts, pass to policy as `target_entropy`. Requires custom trainer hook or wrapping `DiscreteSACPolicy`.

**Current workaround**: `entropy_frac=0.3` (down from 0.5) as a static approximation.

---

## Current plan
Next run: actor-only resets (`reset_freq=100`), critics untouched, `entropy_frac=0.3`.
