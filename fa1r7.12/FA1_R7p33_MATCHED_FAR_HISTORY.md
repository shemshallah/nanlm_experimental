# FA1 R7.33 — multi-horizon matched-history curriculum

**Continuation of R7.32 on `master`, not another language model.** The same
inherited 96D trunk, byte/subword/token/word/sentence attention, 256 heads,
32,640 signed S+/S− pairs, 2,763,520 triangle addresses, sparse apex
experts and descending geometric feedback remain intact. R7.32's 17-group
unified optimizer and all historical moments are preserved unchanged. R7.33
adds a multi-horizon training/evaluation protocol, **no new trainable weights**.
R7.32 already implements a single-window matched-history comparison;
R7.33 reuses that exact causal control without reimplementing it.

## Why this is the next experiment

The earlier 512-update R7.31 research run reduced conditional spelling
bits/byte **6.07557 → 4.01454** and improved next-byte top-5
**29.48% → 58.01%**. Yet correct-book advantage regressed
**−0.00217 → −0.00364 nats/word**. The decoder is learning spelling; we
still have no demonstrated useful *older-book-history* contribution from
geometry. This prior candidate is not recoverable in the current workspace;
its reported metrics are historical, not a new R7.33 training claim.

## Strict matched-history intervention

For every selected training/DEV prediction, choose a donor from a different
book. Keep the gold target, inherited language-model hidden state, selected
sparse concept vertices, all **60 near-memory features** and first **8 near
modalities** exactly as they were. Replace only the older token/byte/sentence
prefix, the **32 far-memory features** and the **4 far modalities** from
the *same* foreign-book donor. The protected recent suffix varies:

| Horizon | Most recent words kept | Bytes kept | Sentence IDs kept |
|---|---:|---:|---:|
| 0 — inherited R7.32 | 8 | 32 | 2 |
| 1 | 16 | 64 | 4 |
| 2 | 24 | 96 | 6 |

This preserves the most recent context and changes distant information.
Because sparse vertex IDs have no temporal ordering, they remain unchanged.
The experiment is a conservative attribution control, not a claim that all
sources of recent information are exactly matched. The actual donor is chosen
independently at each horizon and is always from another book.

## What the new measurements must prove

| Diagnostic | Positive means |
|---|---|
| Matched-far word advantage | Real older history lowers held-out word NLL |
| Matched-far byte advantage | Real older history lowers conditional UTF-8 spelling CE |
| Word geometric specificity | Geometry improves predictions *more* on correct than on wrong older history |
| Byte geometric specificity | The same difference-in-differences holds per scored literal byte |
| Book bootstrap lower 95% bound | Evidence survives sampling uncertainty across DEV books |

Geometric specificity is calculated from four matched predictions:

`(correct_geometry_off − correct_full) − (foreign_geometry_off − foreign_full)`.

Both matched-history advantage and geometry-specific advantage must be
positive, including the lower book-bootstrap bounds, before a candidate
passes the new DEV gate. All original full-word/factor/apex/pair and
conditional byte gates remain mandatory. TEST remains sealed.

DEV reports a result for each horizon and forms the **worst-horizon
per-example** advantage for its overall gate. Neither an easy donor nor a
single successful recency window can conceal a failure at another scale.

TRAIN cycles through the three horizons on successive ablation intervals and
adds word-branch and byte-decoder matched-history margin losses. On ablation
steps, it adds geometry-specific contrast on both pathways, with
geometry-off teachers computed *before* constructing any backward graph to
avoid mutating autograd-saved tensors. No gold future byte enters any
context vector. The byte head still receives teacher-forced earlier bytes;
conditional bits/byte is not raw contiguous-text bits/byte.

## Repeatable research continuation

Requires the intact R7.31 step-32 or R7.32 unpromoted parent checkpoint,
the verified R7.13 ancestor, and the exact disjoint 270/15/15-book corpus.
R7.31 migration appends the R7.32 router group without resetting moments;
R7.32 continuation restores all 17 groups strictly.

```bash
PYTHONPATH=fa1r7.12/source python fa1r7.12/source/fa1_r728_geometric_memory.py \
  --calibrated --matched-far-history \
  --root R713_ANCESTOR --checkpoint R731_OR_R732_UNPROMOTED.pt \
  --corpus-dir SHA_DISJOINT_300_BOOKS --split dev \
  --train 1024 --dev 512 --steps 128 --minibatch 12 --batch 8 \
  --eval-every 16 --ablate-every 4 \
  --matched-far-weight .35 --matched-far-byte-weight .20 \
  --matched-far-geometry-weight .20 --matched-far-byte-geometry-weight .10 \
  --output R733_DEV.json --candidate-checkpoint R733_UNPROMOTED.pt
```

The immutable source checkpoint must be hashed before and after each run.
Save successors separately; never automatically promote. Evaluate unchanged
book-disjoint DEV positions with full/no-geometry/no-apex/no-pair and
matched-far history, including top-1/top-5 and conditional BPB. Once a
candidate passes all DEV gates, run one sealed TEST evaluation.

GitHub Actions source tests are not held-out training results.
