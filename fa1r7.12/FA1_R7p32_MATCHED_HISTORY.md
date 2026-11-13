# FA1 R7.32 — matched-history contrast for the existing layered Diamond

**Training/evaluation control on `master`, not a new network or promoted checkpoint.**

R7.31 already improves conditional UTF-8 spelling on development examples,
but its geometric route may be a context-agnostic correction. The reported
512-update candidate had word NLL 5.24091, spelling bits/byte 4.01454, byte
top-1/top-5 16.97%/58.01%, geometry word gain +0.00596, yet foreign-book
advantage **−0.00364**. These are prior reported candidate metrics, **not
benchmarks of R7.32**. Raw contiguous text bits/byte and full-mixture word
top-5 are still not measured.

## Controlled question

Can the **same next word**, **same recent history**, and **same inherited R7
hidden state** be better predicted when the Diamond receives the *correct
older book history* rather than unrelated older history?

For each prediction, sample another-book donor, keeping the following fixed:

| Fixed | Replaced with a single foreign book donor |
|---|---|
| Next-word and conditional UTF-8 spelling labels | Never replaced |
| Inherited `hidden`, `target`, `book_ids` | Never replaced |
| Eight active point addresses / latest 8 word IDs | First 24 older word IDs |
| Latest 32 literal context bytes | First 96 older context bytes |
| Latest two sentence heads | First six older sentence heads |
| Sixty near-memory features / eight near modal positions | 32 far-memory features / four far modal positions |

This control cannot leak donor targets into query construction: it samples
using only whole-book IDs and does not alter any gold labels. It preserves the
short-range clues that can otherwise make a foreign-history test trivial.
Its point-address scope is deliberately conservative: the original geometric
points are derived partly from recent words, so they must remain stable for
a matched-recent comparison. This isolates the *older* routed signals, not
every possible effect of replacing point selection.

Let `L_on(h)` be a target's NLL with active geometry and history `h`; `L_off`
is computed with geometric gates zeroed. Define

```
geometric_gain(h) = L_off(h) - L_on(h)
context_specific_gain = geometric_gain(correct_old) - geometric_gain(foreign_old)
loss = softplus(8 * (margin - context_specific_gain)) / 8
```

The loss is evaluated per example (not only on a batch-average difference),
on both the 96D word-tree branch and conditional spelling bytes. During
training, both geometry-off controls are detached and computed **before** any
live autograd graph is constructed, avoiding in-place-gate version errors;
both geometry-on histories receive gradients. The pre-existing overall word
loss, causal byte CE, apex/pair ablations, and full foreign-history contrast
remain active. No extra weights or optimizer groups are introduced.

## Development promotion requirements

All prior strict DEV gates continue to apply. In addition, require positive:

1. Matched-history geometric gain of the final normalized word-mixture NLL.
2. Matched-history geometric gain of the 96D word-tree branch.
3. Matched-history geometric gain of the conditional UTF-8 spelling NLL.
4. Whole-book bootstrap 95% lower confidence bounds for **each** of those
   three metrics.

This must be tested on the same book-disjoint DEV positions, with no TEST
consultation for candidate choice. A selected checkpoint remains **unpromoted**;
the protected accepted step 14,416 is never rewritten. The 16 optimizer
groups and their historical moments are retained. When resuming a checkpoint
already marked `FA1_R7_31_LAYERED_DIAMOND_V1`, learned gate values are used
unchanged rather than reset to `--layered-gate-init` and the other warm starts.

## Candidate-only CPU continuation

Requires the *complete* R7.31 512-update candidate file (not the text table),
its R7.13 ancestral tokenizer/tree files, and SHA-disjoint 270/15/15 whole-
book NaNROUTER corpus. None is embedded or downloaded by this source-only
workflow. Resume and write a NEW output path:

```bash
PYTHONPATH=fa1r7.12/source python fa1r7.12/source/fa1_r728_geometric_memory.py \
  --layered --root R713_ANCESTOR --checkpoint R731_512_UNPROMOTED.pt \
  --corpus-dir DISJOINT_300_BOOKS --split dev \
  --train 1024 --dev 512 --steps 256 --minibatch 12 --batch 8 \
  --eval-every 32 --ablate-every 4 --lr 0.0001 \
  --matched-geometry-weight 0.6 --matched-byte-weight 0.25 \
  --matched-margin 0.005 --output R732_DEV.json \
  --candidate-checkpoint R732_UNPROMOTED.pt
```

Report full-word loss, inherited tree top-1/top-5, conditional spelling
bits/byte and byte top-1/top-5, first-byte accuracy, geometry-off contrasts,
all three matched-history gains and per-book intervals. Compare the same
evaluation positions to the R7.31 parent. Run a separate untouched TEST
evaluation **only after DEV selection and with the candidate frozen**.

The checked-in tests cover donor book isolation, target preservation,
recent-history identity, correct gradient signs and full checkpoint gate
continuation. CI validates source and never promotes or manufactures metrics.
