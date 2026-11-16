# FA2 R49 — Matched causal-history specificity probe

**2026-11-16 UTC · `master` / `poincare_mirror` · DEV-only · no promotion**

## Research question

R47/R48 found that reducing or removing reciprocal output emission improves
several *trained-candidate* metrics, but does not improve over the protected
step-2,993 parent. Before widening or extending training, we need to know
whether the existing reciprocal path uses the **correct preceding history**
in a way the inherited transformer does not already provide.

For each fixed next-byte target, the R49 evaluator constructs two equal-length
causal inputs:

1. `true`: the genuine preceding 128 input bytes from a DEV book;
2. `wrong`: 96 older bytes replaced by an interior prefix from a *different*
   DEV book, while the latest 32 true bytes and the next-byte target stay
   identical.

The next target byte is **never** included in either model input. Every
pair is matched across three modes: inherited full reciprocal emission,
internal coupling-only (all outer emission gains zero), and reciprocal
processing disabled. The key per-book contrast is:

`(CE_wrong - CE_true)_reciprocal_on - (CE_wrong - CE_true)_reciprocal_disabled`.

A positive value means reciprocal processing improves correct-history
sensitivity beyond ordinary context processing. It does **not** by itself
identify which particular mirror-pair, triangle or apex addresses supplied
that benefit; an address-specific control is a separate experiment.

## Immutable checkpoints and corpus

| Role | Optimizer step | SHA-256 |
|---|---:|---|
| Protected parent | 2,993 | `74478d2f77b3c2da7978458f3660dffe8aad788938ce4a7ab6eae7cff2de0817` |
| R48 CPU continuation, unpromoted | 3,025 | `36e14da9676690595c523456c391fca86977bf9799ce135508223f54e49e013e` |

Corpus: original 38-book byte/EOS split (12 TRAIN, 12 DEV, 14 sealed
TEST). The DEV binary SHA-256 is
`0314f5cb62b0e06581a17ad98f3f3e93b7d0f8d285a65253adfc128b481f54cb`.
Both checkpoints use the exact matching token and split manifests.

## DEV measurements

These are **16 causal windows per each of 12 independent DEV books**,
**192 next-byte targets per mode**, with seed 2026. They are not exhaustive
book scores. We used 2,000 book-cluster bootstrap replicates and a two-sided
95% Bonferroni correction across the four preregistered reciprocal-specific
contrasts (two enabled modes × two checkpoints).

| Checkpoint | Mode | True-history CE ↓ | Correct-history advantage ↑ | Reciprocal-specific increment ↑ | Adjusted interval for increment |
|---|---|---:|---:|---:|---|
| Step 2,993 | Full | 2.474050 | +0.012146 | +0.002956 | [−0.002261, +0.009031] |
| Step 2,993 | Coupling only | 2.474509 | +0.008643 | −0.000547 | [−0.002188, +0.000010] |
| Step 2,993 | Reciprocal disabled | 2.474521 | +0.009190 | — | — |
| Step 3,025 | Full | 2.485214 | +0.000259 | +0.001934 | [−0.001910, +0.005634] |
| Step 3,025 | Coupling only | 2.484885 | −0.001506 | +0.000169 | [−0.000020, +0.000675] |
| Step 3,025 | Reciprocal disabled | 2.485055 | −0.001675 | — | — |

The CE values are in **nats per next-byte target**. The `true` inputs' BPB
can be computed as `CE / ln(2)` because each sampled target is one literal
byte; these scores cannot be compared directly to R47's different 12,288
target-position selection. Top-1 and top-5 for every arm are preserved in
the raw report.

**Result:** none of the four reciprocal-specific intervals has a positive
lower bound. Parent full emission has a positive point estimate, but is not
statistically established. On this small DEV sample, the R48 candidate's
correct-history advantage is near zero in full mode and negative in
coupling-only mode. These observations do not establish that training erased
history information, because their uncertainty and sample size do not
support that stronger claim.

**Decision: no evidence sufficient for promotion.** Retain the protected
parent; preserve R48 only as an explicitly unpromoted recovery artifact.
No sealed TEST book was opened and no model weights or Adam histories were
modified by R49.

## Reproduce and verify

```bash
cd poincare_mirror
python scripts/probe_r49_causal_history.py \
  --parent /path/to/step2993.pt \
  --candidate /path/to/step3025_unpromoted.pt \
  --data-dir /path/to/verified_gutenberg38_bytes \
  --context 128 --near 32 --windows-per-book 16 \
  --batch 4 --bootstrap 2000 --seed 2026 --threads 2 \
  --output /path/to/r49_dev.json
python -m unittest discover -s tests -p 'test_r49_causal_history.py' -v
```

The evaluator and its tests verify disjoint donor book selection, preservation
of the genuine recent suffix, exclusion of the target from visible inputs,
deterministic sampling and properly adjusted paired-book bootstrap intervals.
The JSON report is at
`results/FA2_R49_causal_history_DEV_20261116.json`.

## Next research steps

1. Re-run the history test with **longer causal windows** and identical target
   positions to identify the context distance at which geometry becomes useful.
2. Add an exact **address-shuffle / phase-rail control** inside pair and
   triangle routing, preserving all nongeometric operations.
3. If the true-address path beats those controls, train a bounded
   **correct-history-specific contrastive loss on TRAIN only** within the
   existing model and single Adam lineage, with exact zero-gate fallback.
4. Require independent-book next/+2/+3 likelihood, top-1/top-5, non-space
   improvement, and long-context gates before sealed TEST and any promotion.

This R49 research path stays entirely separate from FA1 and Terminal-RSI.
