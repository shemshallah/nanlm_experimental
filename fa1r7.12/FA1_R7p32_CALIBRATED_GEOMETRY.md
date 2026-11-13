# FA1 R7.32 — confidence-calibrated Diamond (source continuation)

**Branch:** `master` only. **One model, one unified optimizer.**

## Motivation from R7.31 DEV evidence

R7.31's first selected step-32 DEV candidate improved full-word NLL
5.46501 → 5.46412 and conditional spelling 8.2536 → 7.7693 bits per
literal UTF-8 byte, but geometry hurt the byte decoder by roughly 0.00444
nats/byte and unrelated-book history remained too competitive. The step-48
decoder reached 7.4931 bits/byte but regressed full-word NLL to 5.46642.
Neither candidate passed strict geometric-history promotion criteria.

R7.32 addresses two independent issues:

1. **Adaptive geometric trust.** Keep the inherited 256 vertices, 32,640
   signed mirror pairs, 2,763,520 exact triangle addresses, apex/tetrahedron
   specialists and two-way descent. After pair→point descent and apex
   integration, learn separate, causal per-example *word* and *byte* trust
   factors. Each is in (0, 2), initialized to **exactly 1**. A memory read
   that harms the byte or word prediction can be suppressed; useful reads
   may be amplified. The transformer spelling head and all inherited
   parameters, addresses and optimizer moments are unchanged on migration.
   The same unified optimizer grows from 16 to 17 parameter groups; this
   does not instantiate a second optimizer or reset a learned moment.
2. **Selection based on admissibility.** Pick the lowest word NLL among
   candidates passing the *full* DEV promotion gate first. If none passes,
   use a separate tier for positive word geometry, history, apex, pair,
   byte geometry and improved word NLL/byte BPB. Only if both fail, report
   the lowest-word-loss diagnostic candidate. This is not an automatic
   promotion and does not access the 15 sealed TEST books.
3. **Unconfounded ablation.** A prior "geometry off" control also disabled
   the new byte/subword/token/word/sentence context loop. That loop carries
   language information even without geometry. R7.32 keeps it active during
   geometry-off evaluation and reports a separate language-context-off
   diagnostic. Historical R7.31 geometry gains and new R7.32 geometry gains
   must be recalculated on matched examples before they are compared.

The strict gate also requires the conditional spelling bits/byte to improve
over its source checkpoint and top-1/top-5 not to regress. The report now
records the chosen tier, best word-loss step and best byte-BPB step
separately; these are not interchangeable models. A report produced without
writing a `.pt` restores the selected model and optimizer for its final
measurements, fixing a previous mismatch between reported and measured steps.

## Metric definitions

- `conditional_spelling_bits_per_byte`: literal-byte teacher-forced
  cross-entropy / (actual scored UTF-8 bytes × ln(2)), excluding EOW.
- `conditional_spelling_top1`, `conditional_spelling_top5`: exact next-byte
  accuracy per literal UTF-8 byte, with the previous *gold* byte supplied.
- `conditional_spelling_eow_nll_per_word`: scored separately.
- `conditional_byte_geometry_gain_nats_per_byte`: geometry-off byte loss
  minus full byte loss, **weighted by scored bytes**; positive is useful.
- `correct_book_advantage_nats_per_word`: unrelated-book NLL minus
  correct-book NLL, using matched targets; positive is useful.
- `raw_contiguous_text_bits_per_byte`: `null`. This model receives word
  boundaries; conditional spelling BPB must never be called raw-text BPB.
- `prediction_metrics`: inherited normalized word-tree top-1/top-5 and
  first-byte metrics where supported, not a fabricated full-mixture top-5.

## Reproduction / checkpoint safety

The R7.31 step-32 `.pt`, full tokenizer/tree ancestor, and exact 270/15/15
book split are required to claim a resumed R7.32 training comparison. CI
checks only source contracts; it does **not** ship protected checkpoints or
private datasets. Use the same parent file and DEV split for both conditions.

```bash
PYTHONPATH=fa1r7.12/source python fa1r7.12/source/fa1_r728_geometric_memory.py \
  --calibrated --root R713_ANCESTOR --checkpoint R731_STEP32_UNPROMOTED.pt \
  --corpus-dir DISJOINT_300_BOOKS --split dev \
  --train 1024 --dev 512 --steps 96 --minibatch 12 --batch 8 \
  --eval-every 16 --ablate-every 4 --layered-gate-init .12 \
  --byte-ar-weight .35 --byte-geometry-weight .2 \
  --output R732_DEV.json --candidate-checkpoint R732_UNPROMOTED.pt
```

The accepted 14,416-step model is never overwritten. Re-run full-geometry,
no-geometry, no-apex, no-pair and wrong-history DEV ablations, then evaluate
one guarded successor on untouched TEST books only after the full DEV gate.
Do not promote merely because a synthetic test or the source CI passes.
