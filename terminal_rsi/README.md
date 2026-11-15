# Standalone Terminal-RSI research track

**Separate from FA1/FA2, Poincaré Mirror LM, and all language-model checkpoints.**

This directory implements the independently testable *mechanics* of
[Terminal-RSI (Yang et al., 2026)](https://arxiv.org/abs/2610.22192):

1. A trusted verifier rejects unrecognized or incorrect actions.
2. A successful public donor trajectory is replayed under a different harness
   and independently verified before it is admitted to RSR.
3. RSR learns supervised actions while preserving the critic.
4. Four critic-only calibration updates leave actor/Adam tensors unchanged.
5. RPR samples the current policy and uses verified rewards with one-step
   turn-aware GAE and a single-epoch on-policy policy-gradient update (the
   ratio-1 limit of clipped PPO; no multiple-epoch PPO claim).
6. Repeat RSR/RPR without resetting actor or optimizer state.
7. Test held-out task IDs separately; never automatically promote candidates.

No shell execution, unconstrained self-rewrite, pretrained model, external
checkpoint, Poincaré geometric weight, or FA2 corpus is used. The toy actor is
a lightweight 10-way categorical policy with persistent Adam moments.

## Reproduce

Requires Python 3.11+ and NumPy.

```bash
python -m unittest discover -s terminal_rsi/tests -v
python -m terminal_rsi.run --sft-updates 200 --rpr-updates 100 \
  --out terminal_rsi/reports/standalone_toy.json
python -m terminal_rsi.benchmark --out terminal_rsi/reports/standalone_multiseed.json
```

Optional `--candidate <path>` writes a separate **unpromoted** local JSON
with actor, critic, optimizer moments, and training step. Do not commit trained
weights into GitHub source. The source-only report carries no checkpoint.

## What the metrics mean

- `strict_success_rate` / `top1`: deterministic exact one-action task success.
- `top5`: fraction of correct actions included among five proposed actions.
- `nats_per_action`, `bits_per_action`: probability assigned to the one
  verified output action; **not language bits/byte or bits/BPE token**.
- `sampled_train_*`: genuinely sampled on-policy **training** actions;
  these counts are not held-out terminal-task successes.
- `tasks_per_second`: wall-clock *task scoring* rate, not token throughput.

The 1,000 synthetic digit-command tasks (800 training, 200 held-out) have
disjoint task IDs **but share task families**; successful generalization on
them would not establish Terminal-Bench competence. A real Terminal-RSI result
requires an independently executed terminal-task dataset, authenticated
verifier receipts, comparison to nonrecursive baselines, multiple seeds, and
strict success/partial-credit metrics. Until then `benchmark_eligible=false`.

The former experiment under `poincare_mirror/poincare_rsi_adapter/` is a **historical
optional model-specific adapter**, not this standalone track. Its results
must never be combined with or presented as Terminal-RSI benchmarks.

For source-verified low-BPC prediction experiments on actual NaNROUTER books,
the agent now owns an **optional v10.2-inspired byte specialist** at
[`byte_specialist/`](byte_specialist/README.md). It contributes byte-context
features and vocabulary-aware predictions without replacing this actor or
importing FA1/FA2. Its separate book BPC metrics and terminal-task scores
remain explicitly different evaluation tracks.
