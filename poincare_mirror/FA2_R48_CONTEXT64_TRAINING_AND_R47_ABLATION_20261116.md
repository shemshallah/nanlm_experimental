# FA2 R48 — R47 emission reproduced; geometry-only CPU continuation rejected

**2026-11-16 UTC · `master` / `poincare_mirror` · DEV-only · no promotion**

## Question and provenance

Can frozen-weight per-block reciprocal emission rescue the current geometric
training objective? R47's six predefined schedules are `full`, `coupling_only`,
`quarter`, `half`, `early_only`, and `late_only`. All retain the existing
pair/triangle/apex internal coupling. There are no new emission parameters.

The independently recovered parent is FA2 geometry warm-up step **2,993**,
SHA-256 `74478d2f77b3c2da7978458f3660dffe8aad788938ce4a7ab6eae7cff2de0817`.
The exact 38-book corpus has 12 TRAIN, 12 DEV and 14 untouched TEST books.
The DEV binary is SHA-256
`0314f5cb62b0e06581a17ad98f3f3e93b7d0f8d285a65253adfc128b481f54cb`.

The first *read-only* six-way R47 sweep on the recovered step-2,993 parent
exactly reproduced the previously reported parent next-byte CE **2.383392**,
bits/byte **3.438508** and non-space CE **2.650380**. Its lowest measured
BPB was `early_only` **3.438471** (a **0.000037** BPB difference) but its
Bonferroni-adjusted book-resampled CE interval against `full` crosses zero.
No emission schedule establishes a benefit on this parent.

The original R44 candidate trained at step 3,025, SHA-256
`fda6558b290924d0dfa93c3d270294ad37b930787c167128b6ca1554b5a592cb`,
was **not** available in the recovered files. The continuation below is a
new, genuinely executed training run, **not** a reproduction of that
historical candidate.

## Bounded CPU continuation

An initial 128-token-context attempt was killed by the workspace memory
limit after its first update, before any new checkpoint was saved. The
independent parent remained byte-identical. A subsequent geometry-only
continuation used context **64** to fit the available memory, while retaining
the original 192-dimensional, six-block model, the same byte vocabulary and
one optimizer lineage.

Starting at step 2,993, the continuation completed 1 update followed by
31 resumed updates to step **3,025**: batch 1, learning rate `0.0001`,
train-only non-space weight `1.15`, R44 geometric residual auxiliary loss
weight `0.25` and fixed auxiliary logit scale `8`. Validation was a small
training diagnostic; admission used a separate, matched book-cluster DEV
evaluation at context 128.

Lineage audit passed:

- **194/194** inherited Adam states retained.
- **63/63** frozen language-backbone parameter sets bitwise unchanged.
- **63/63** frozen language-backbone Adam state sets bitwise unchanged.
- **131** geometric parameter sets changed and their Adam steps advanced.
- No independent replacement model, optimizer reset, new branch or TEST tuning.

New *unpromoted* candidate SHA-256:
`36e14da9676690595c523456c391fca86977bf9799ce135508223f54e49e013e`.

## New measured DEV comparison

Each schedule and the protected parent saw the same **12 book-disjoint DEV
books × 8 sampled 128-token windows/book = 12,288 predictions**. These are
sampled windows, **not exhaustive full-book scoring**. We resampled whole
books as clusters 2,000 times, correcting six candidate-versus-parent
contrasts within each metric (two-sided Bonferroni).

| Model | +1 CE ↓ | BPB ↓ | Top-1 ↑ | Top-5 ↑ | +2 CE ↓ | +3 CE ↓ | Non-space CE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Protected step 2,993 | **2.383392** | **3.438508** | **35.70%** | **69.88%** | **2.926668** | 3.106511 | **2.650380** |
| Trained full | 2.387549 | 3.444506 | 35.60% | 69.71% | 2.928404 | **3.104670** | 2.655754 |
| Trained coupling-only | **2.387159** | **3.443942** | 35.59% | **69.76%** | 2.928318 | 3.104795 | **2.655236** |
| Trained quarter | 2.387562 | 3.444523 | 35.57% | 69.74% | **2.928233** | 3.104816 | 2.655588 |
| Trained half | 2.387795 | 3.444860 | 35.57% | 69.73% | 2.928342 | 3.104783 | 2.655857 |
| Trained early-only | 2.387515 | 3.444456 | **35.60%** | 69.72% | 2.928302 | 3.104690 | 2.655745 |
| Trained late-only | 2.387299 | 3.444145 | 35.59% | 69.75% | 2.928364 | 3.104823 | 2.655377 |

The best trained next-byte configuration, `coupling_only`, is still **worse**
than the protected parent by 0.003767 nats/byte, or **0.005434 BPB**. Its
candidate-favoring adjusted CE gain interval is
**[−0.008532, −0.000151] nats/byte**. Coupling-only is modestly better than
trained full emission on point estimates (+0.000391 nats/byte), but its
adjusted within-candidate interval [−0.000329, +0.001087] also crosses zero.
Non-space CE and +2 loss regress against the parent. A small +3 improvement
does not offset the regression in next-byte likelihood.

**Decision: 0/6 schedules pass the admission screen. R48 candidate rejected.**
No sealed TEST was opened; the step-2,993 parent and accepted ancestry remain
protected. Do not promote the trained checkpoint or treat the result as
evidence that geometric memory already improves prediction.

## Next experiment: causal specificity before further scaling

The immediate architectural priority is to test whether a pair/triangle/apex
read depends on *the correct preceding document history*. On a fixed DEV
window, compare genuine history against equal-length wrong-book history,
shuffled addresses and zero coupling; require positive correct-history-specific
next-byte and non-space likelihood, without future-token access. Train any
new target-free trust calibration only inside the same inherited model and
optimizer, with an exact zero-gate fallback. Reassess emission placement only
after geometric information itself passes these controls.

## Reproducible records

- `results/FA2_R47_step2993_emission_DEV_20261116.json`: recovered parent,
  fixed-weight six-way emission sweep.
- `results/FA2_R48_context64_step3025_emission_DEV_20261116.json`: full
  six-way candidate versus protected parent, per-book adjusted intervals,
  all horizons and explicit byte denominators.
- `results/FA2_R48_context64_step3025_lineage_20261116.json`: exact lineage
  audit, hashes and optimizer continuity.
- `scripts/probe_r47_selective_emission.py`: matching DEV evaluator;
  `tests/test_r47_selective_emission.py`: causal, restoration, bootstrap and
  six-arm multiplicity regressions.

The full trainable candidate retains its model, one optimizer, RNG state,
growth history and corpus manifest. Its checksum above identifies the exact
checkpoint. No FA1 or Terminal-RSI files were changed.
