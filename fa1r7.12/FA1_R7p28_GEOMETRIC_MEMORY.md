# FA1 R7.28 — Training geometric storage, not just geometric activation

Development branch: **`master` only**. This continues the R7.27 unified
checkpoint and optimizer. The accepted R7 ancestor remains step **14,416**;
trained R7.28 states are **unpromoted** research candidates.

## Failure measured at R7.27

The previous orientation/pair/face/tetra/apex paths were mathematically live
but did not supply a positive geometric *causal ablation* on held-out books.
The geometric specialists were especially weak: `meta_gate=0` prevented their
per-address values from changing predictions. Distinct activation and an
interesting curvature trace are not evidence of predictive utility.

## Storage and write/read mechanism retained inside one model

The model has 256 inherited heads, 32,640 signed mirror pairs, 2,763,520
exact triangle addresses, and 174,792,640 possible tetrahedra (70 activated
among the eight selected points). The original R7.23 address `triple_code`
and the R7.24 signed face-code embeddings `face_plus` / `face_minus` are
**persistent learnable parameters**. Training produces explicit gradient and
row-change counts for those tensors. Four of the 256 apex specialists are
selected causally per prediction and feed back into triangle -> pair -> point
reconstruction. Both signs S+/S− and the tetrahedral curvature/closure path
remain active; no alternate network is trained independently.

The R7.28 trainer adds **per-example target-matched causal contrasts**:

1. Supervise the Diamond's actual next-vocabulary-word tree likelihood and
   its auxiliary first-byte head, not activation magnitude.
2. Contrast the correct *preceding* document context with a donor from a
   different book. Keep the true label and inherited R7 memory unchanged.
3. Train the geometric code to beat the otherwise identical model with its
   edge/face/tetra/meta channels disabled. This directly rewards informative
   signed geometric storage.
4. Require the complete apex and descending pair->point route to outperform
   their separate causal ablations on the training target.
5. Open the already-trained Diamond probability bridge only on the candidate;
   preserve inherited trunk, memory, and optimizer moments.

The 13 optimizer groups still execute through **one** R7.27 optimizer object,
retaining the exact historical AdaGrad-like first group and AdamW groups.
Only Diamond and geometry parameters receive new gradients in this pilot.

## Benchmark protocol

- Source: NaNROUTER `short_stories-0004.tar.zst`, SHA-256
  `d6ce3a23757e85fe1b9487ef984a5ba81db9c2dc6e962ac7c5df2960480e0054`.
- 216 TRAIN books, 12 DEV books, 12 **sealed TEST** books; disjoint book SHA-256s.
- Each DEV step compares identical target positions for baseline, geometry-on,
  geometry-off, correct/foreign history, apex-off, and descending pair-off.
- Report full-model next-word NLL, inherited-tree normalized word top-1/top-5,
  auxiliary first-byte top-1/top-5, book-cluster bootstrap intervals, exact
  address writes, and context-ablation losses.
- The R7 full decoder does **not** yet provide all-vocabulary normalized top-5
  nor raw-byte autoregressive BPB. First-byte-of-word loss is **not** BPB.
- Passing a development gate requires full word NLL improvement, positive
  geometry-on causal effect, positive correct-book advantage, positive
  apex and pair effects, and strictly positive *book-level lower 95% bounds*
  for geometric and document-history improvements. No TEST peeking.

## Reproduction

The real-weight local command is:

```bash
python fa1r7.12/source/fa1_r728_geometric_memory.py \
  --root PATH_TO_RESTORED_R713_ANCESTOR \
  --checkpoint PATH_TO_R727_UNPROMOTED_CHECKPOINT.pt \
  --corpus-dir PATH_TO_NANROUTER_BOOK_SPLIT \
  --output PATH_TO_DEVELOPMENT_REPORT.json \
  --train 768 --dev 256 --steps 96 --minibatch 16 \
  --batch 8 --eval-every 16 --ablate-every 3 --threads 2 \
  --candidate-checkpoint PATH_TO_NEW_UNPROMOTED_R728.pt
```

`fa1-r728-geometric-memory.yml` runs source and causal-control tests on GitHub
Actions `master`. GitHub runners cannot restore the private R7.27 Library
checkpoint unless a verified checkpoint asset is made available. A passing
source workflow is therefore **not** represented as a full neural benchmark.

## Measured CPU results — 2026-11-13

Training used the authentic R7.27 unpromoted checkpoint (SHA-256
`f24ccc9c2a906504ee8d26c04b76e53f34010d34ccc61408e9c88ac99920c8ae`)
and the same single hybrid optimizer, with **96 attempted R7.28 steps** and
the DEV-best candidate at step **32**. Its recoverable research checkpoint
has SHA-256
`91e3220f9167984bf6d529b348bea610fadc05c5d4950ccdf8fe63911e2aae5b`.

| Fixed metric | Untrained R7.27 | R7.28 step-32 | Interpretation |
| --- | ---: | ---: | --- |
| DEV word NLL (256 positions) | 5.259902 | 5.257688 | Small DEV improvement |
| DEV geometry-on minus geometry-off gain (nats/word) | -0.000741 | -0.000925 | **Still negative** |
| DEV correct-book advantage (nats/word) | -0.001041 | +0.002458 | Positive point estimate; book CI crosses zero |
| TEST word NLL (512 positions) | **5.440382** | 5.444346 | **Regression, reject** |
| TEST first-byte top-1 | **13.09%** | 11.33% | Regression |
| TEST first-byte top-5 | 37.70% | **42.77%** | +5.08 percentage points |
| TEST inherited normalized word top-5 | 33.98% | 33.98% | Trunk frozen |
| TEST correct-book advantage (nats/word) | -0.000866 | +0.000830 | Tiny, uncertain |

The TEST comparison used 512 matched next-word positions across **six** books.
Paired full-NLL candidate gain was **-0.0039629 nats/word** with book-cluster
95% bootstrap CI **[-0.009269, +0.000508]**. This is not an improvement.
The candidate changed approximately 34,552 signed S+ triangle-code rows,
34,560 S− rows and 59,038 original R7.23 triangle-code rows. Those are
learned persistent parameters, but *writes alone do not establish predictive
causality*. Exact-address rows are also sparsely shared across unseen books.

An exploratory signed-gate scan on **DEV only** preferred a -0.5 scale versus
the learned positive orientation, with a tiny +0.000191 nats/word gain over
geometry-off. Its bootstrap interval includes zero. **It was not promoted and
was not applied to the saved R7.28 candidate.** This points to miscalibrated
geometric readout orientation and the need for context-dependent trust.

The 512 TEST positions were used for this fixed, final comparison only; they
cannot be reused to select a new architecture. Prior training-book overlap
with the historical ancestor checkpoint is not independently established.

Sources: `source/fa1_r728_geometric_memory.py`,
`source/fa1_r728_final_benchmark.py`, and
`source/fa1_r728_gate_sweep.py`. Frozen JSON receipts are in `reports/`.
