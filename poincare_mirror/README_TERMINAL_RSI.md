# Terminal-RSI adaptation — verified actor/critic continuation

**Research implementation, not a reproduction of the 27B Terminal-RSI results.**
Reference: [Yang et al., arXiv:2610.22192](https://arxiv.org/abs/2610.22192),
[reference source](https://github.com/jyyang26/Terminal-RSI).

## Same Poincaré model, additional training discipline

The original `MirrorLM` still owns the byte hierarchy, geometric reciprocal
self-attention loops, signed S+/S− pairs, triangle/tetrahedral pathways and
long-context memory. `terminal_rsi/torch_adapter.py` appends one initially
zero-valued critic projection to its final hidden state. It adds a separate
**parameter group to the same Adam optimizer**, retaining all original groups
and moments. The actor's state dict keys and logits are unchanged at migration.

```
Existing byte/subword/word/sentence geometry + mirrored MoE actor
                    |                        ^
               hidden states          supervised / PPO gradient
                    |                        |
                    +--> inherited critic --+  (new value head only)
                                           |
          trusted replay + external verifier receipt
                                           |
                     RSR actor SFT <--> RPR turn-aware PPO
                                           |
                 independent-book + nonspace + history gate
```

The paper's recursive phases are **RSR-1 → RPR-1 → RSR-2 → RPR-2**. The
implementation preserves this transition order and requires at least four
critic-only warmup updates between an RSR phase and an RPR phase. RSR updates
must not modify critic weights. It only learns from assistant actions in a
successful public trajectory that was independently replayed **under a fresh
target harness** and approved by an external trusted verifier. Internal
instructions, private tool traces and arbitrary command execution are not
permitted by the adapter.

RPR implements the paper's **turn-boundary skip-observation GAE**. Tool outputs
remain visible as causal context but do not form advantage-recursion nodes or
contribute action loss. Every artificial chunk boundary must provide the next
generated action's value, advantage and turn ID; terminal states never
bootstrap. Family Dense Reward uses frozen historical p90 semantic-check
divisors with 82 fallback. Success-Prioritized Reward preserves the strict
success tier. Calibration adaptation must be carried out offline.

## Training and source verification

```bash
cd poincare_mirror
python -m unittest discover -s tests -p 'test_terminal_rsi*.py' -v
python scripts/terminal_rsi_pilot.py --sft-steps 80 --ppo-steps 32 \
  --out reports/terminal_rsi_toy_pilot.json
```

The bounded pilot **really updates** one 32D/2-block geometric `MirrorLM`
using verified one-byte successor tasks, actor-only SFT, four critic warmup
updates, and on-policy PPO from sampled actions checked by a deterministic
external task verifier. The held-out letters are disjoint from training
letters. Output reports baseline, after-RSR and after-RPR top-1/top-5,
nats/byte, bits/byte, bytes/token (1 for byte tokens), throughput, and
sampled strict success counts.
The RPR actor uses a stage-specific learning-rate multiplier (`--rpr-actor-lr-scale`,
default 0.1) while preserving all accumulated optimizer moments.

These are **toy task controls**, not NaNROUTER books or terminal-bench scores.
No original R38 model weights are trained or replaced by this pilot. Even a
positive tiny pilot does not warrant promotion.

## Required real continuation prerequisites

1. Recover the complete FA2 R38 **step-4,057** checkpoint, original Adam
   states/RNG, source revision **and matching successor-memory module**.
   The GitHub `master` repository does not yet contain that R38 module.
   Never substitute older R30 weights to manufacture R38 training.
2. Verify exact checkpoint hash, lineage, model keys, config, optimizer groups
   and moment values. Preserve an untouched parent, attach the critic and
   write a separate unpromoted *candidate* with original keys intact.
3. Collect actually executed terminal task trajectories, two independent
   harnesses and trusted receipts (not self-reported outcomes). Construct
   byte-aligned teacher-forced SFT records and **sampled** PPO rollouts.
4. Train on task-disjoint training data and evaluate on previously unseen
   terminal tasks and NaNROUTER **book-disjoint** DEV windows with fixed seeds.
5. Before promotion, require independent book BPB improvement, no primary,
   future-horizon or **non-space CE** regressions, nondecreasing top-1/top-5,
   correct-history superiority over mismatched book history, no terminal
   strict-success regression, and no reduction in verified coverage. Use
   whole-book confidence intervals for nonspace improvement and leave sealed
   test books untouched until model selection is final.

`terminal_rsi.core.promote` is a fail-closed preliminary gate. It does not
replace the existing R38 whole-book confidence and leakage controls. No module
in this directory automatically invokes tools, edits source, publishes an
artifact, overwrites a checkpoint, or promotes a model.
