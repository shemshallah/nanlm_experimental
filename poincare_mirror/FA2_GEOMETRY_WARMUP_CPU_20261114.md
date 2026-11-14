# FA2 — Reciprocal-geometry CPU continuation (2026-11-14)

## Status: trainable; research candidate, NOT promoted

This work uses only the existing `poincare_mirror/poincare_frontier.py`
`MirrorLM` on `master`. It does not alter FA1 or replace the inherited
language head. The accepted, complete FA2 R30 step-2,801 checkpoint is
unchanged and remains the canonical fallback.

The source now supports `train --geometry-only`, which trains the existing
hierarchy, reciprocal tokenizer, and geometric MoE modules while retaining
**every inherited language parameter and Adam state in the same optimizer**.
Its second repair removes an unintended second cold gate at the
microtransformer *input*. The returned geometric update is still multiplied
by one cold gate, so freshly migrated logits remain exactly identical.

## Verified provenance

- Original checkpoint: `poincare_mirror_latest.pt` SHA-256
  `9ad6a368a9e68f4c1ccff366bee88047f7475f93ccb4d1da1c999db7bddee902`.
- R30: step **2,801**, **21,304,992** trained parameters, **63** Adam states
  (126 moment tensors); 257-symbol UTF-8 byte/EOS vocabulary.
- Recovered exact 38-book dataset: 12 TRAIN, 12 DEV, 14 sealed TEST books.
  Manifest SHA-256 `a7bb4406b1606def1be2aefdce0c2a6962999459eced6bc19b460d7ce55e96fc`.
  Validation binary SHA-256 `0314f5cb62b0e06581a17ad98f3f3e93b7d0f8d285a65253adfc128b481f54cb`.
- Function-preserving migration to **23,472,023** parameters:
  192D, six blocks, signed S+/S− mirror experts, sparse 256-point / 32,640
  pair / triangle / apex meta geometry; byte/pair/quad/word/sentence
  reciprocal scales, simplex, curvature, and holonomy enabled at rank 8.
  Three-horizon maximum logit drift **0.0**; **126** inherited moments verified.
- The repaired candidate completed **192** new Adam steps, finishing at
  **2,993**. Stage one: 64 updates, context 128, LR 0.0003. Stage two:
  128 updates, context 128, LR 0.0001. Batch 1, CPU two threads, no optimizer reset.
- Actual checkpoint audit after stage one: **all 67 inherited state tensors
  unchanged** and **all 189 inherited optimizer fields unchanged**, including
  steps and both moment arrays. New geometry owns 2,167,031 trainable parameters.
- Full CPU suite: **92 tests pass, 2 skipped**; includes a new single-gate
  sensitivity regression and book-bootstrap/corpus-boundary tests.

## Matched DEV evaluation (no promotion claim)

The first table uses 64 deterministic 128-byte windows (8,192 target bytes),
with **exactly the same** seed and target positions on both checkpoints.
All cross-entropies are in nats per literal byte/EOS prediction.

| Metric | R30 step 2,801 | FA2 geometry step 2,993 |
|:--|--:|--:|
| +1 CE ↓ | 2.198100 | **2.193799** |
| +1 bits/byte ↓ | 3.171188 | **3.164984** |
| +1 top-1 ↑ | 38.8550% | **38.9893%** |
| +1 top-5 ↑ | 73.7183% | **73.7549%** |
| +2 CE ↓ | 2.803330 | **2.799448** |
| +2 top-1 ↑ | **23.7793%** | 23.6450% |
| +2 top-5 ↑ | 56.1157% | **56.3843%** |
| +3 CE ↓ | 3.023243 | **3.018881** |
| +3 top-1 ↑ | 17.9077% | **18.0054%** |
| +3 top-5 ↑ | 48.7305% | **48.8159%** |

At context 256, another 8,192 aligned DEV byte positions gave:
next-byte CE 2.264797 → **2.261108**; top-1 37.2437% → **37.4512%**,
but top-5 **72.0947% → 72.0093%** (a small regression). The +2 and
+3 losses improved from 2.860359/3.065254 to 2.859123/3.062670.

The initial unrestricted eight-update pilot also ran successfully, but
regressed at context 128 (CE 2.198100 → 2.230623). An earlier 64-update
warm-up before the single-gate repair did not pass the held-out gate.
Neither earlier pilot replaced the canonical R30 model.

## Independent-book bootstrap: promotion REJECTED

Using `scripts/compare_geometry_books.py`, we measured **eight separate
128-byte windows per each of 12 independently held-out DEV books**:
12,288 target bytes per model, 2,000 paired bootstrap resamples of whole
books. The book weighting and sampled positions differ from the window-level
table above; these are complementary estimates.

| Book-balanced measure | R30 | Candidate | Gain (candidate better if positive) | Book 95% CI |
|:--|--:|--:|--:|:--|
| +1 CE | 2.384847 | 2.383392 | +0.001455 | **[-0.002286, +0.004738]** |
| +2 CE | 2.926105 | 2.926668 | -0.000563 | [-0.005079, +0.003223] |
| +3 CE | 3.109404 | 3.106511 | +0.002893 | [+0.000859, +0.004665] |
| +1 top-1 | 35.7666% | 35.7015% | -0.0651 points | [-0.3499, +0.2279] points |
| +1 top-5 | 70.3613% | 69.8812% | -0.4801 points | [-1.4813, +0.2279] points |

The isolated reciprocal-enabled versus reciprocal-disabled +1 CE benefit
was only **+0.000028 nats/byte**, with 95% book interval
**[-0.000300, +0.000382]**. It is **not demonstrated to be positive**.
The complete geometry also lacks sufficient proof of independent predictive
value. Accordingly, `development_gate_pass=false` and **no checkpoint
promotion**. The sealed TEST split remains untouched for model selection.

An aligned context-128 versus context-1024 probe on the **same 512 DEV target
bytes** also showed a long-context warning: at context 1,024, R30 BPB was
3.318281 while the candidate was **3.331383** (higher is worse). The
long-minus-short penalty grew from 0.080937 to 0.105611 BPB. The small
sample does not establish a general long-context result, but it reinforces
the decision not to promote.

## Reproduce from the canonical R30 copy

```bash
python poincare_frontier.py grow --run-dir candidate \
  --symbiotic-rank 8 --hierarchy-rank 8 --micro-meta \
  --simplex-transfer --triangle-curvature --holonomy-retrieval \
  --reciprocal-scales --threads 2 --max-logit-drift 1e-5

python poincare_frontier.py train --run-dir candidate --data-dir data \
  --steps 64 --batch 1 --train-context 128 --eval-context 128 \
  --geometry-only --lr 0.0003 --cpu --cpu-threads 2
python poincare_frontier.py train --run-dir candidate --data-dir data \
  --steps 128 --batch 1 --train-context 128 --eval-context 128 \
  --geometry-only --lr 0.0001 --cpu --cpu-threads 2

python scripts/compare_geometry_books.py --before canonical/latest.pt \
  --after candidate/latest.pt --data-dir data --split valid --context 128 \
  --windows-per-book 8 --bootstrap 2000 --seed 2026 --threads 2
```

The next rational experiment is longer training and separately controlled
meta/reciprocal gates on more independently held-out NaNROUTER books. Test
the isolated geometric contribution by whole-book confidence intervals
before further widening or promotion.
