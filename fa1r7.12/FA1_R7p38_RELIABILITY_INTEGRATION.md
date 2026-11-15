# FA1 R7.38 — causal reliability source recovery and next CPU gate

**Status:** source-integrated on `master`, local tests only; **unpromoted**.
No new complete trained checkpoint, independent-book prediction benchmark, or
sealed TEST evaluation is claimed by this integration.

## Why this path needs direct causal proof

The reported 24-update R7.38 run selected update 1. On 15 DEV books, word NLL
changed 4.965899 → 4.965917, conditional spelling BPB 5.4360 → 5.4207,
conditional spelling top-1 6.03% → 6.03%, top-5 38.52% → 38.75%, and
first-byte top-1/top-5 remained 10.83%/43.33%. The isolated word benefit of
the reliability control was effectively zero; the strict gate failed.
These are **results reported before this integration**, not rerun or newly
independently verified. The earlier R7.38 candidate checkpoint/source was not
located under its expected names in the available saved files or GitHub
`master`. Therefore this patch must not be represented as bit-identical to the
unrecovered candidate.

## In-place source implementation

`fa1_r738_reliability.py` subclasses the existing R7.35 triangle writer
without replacing its parameters. Its bounded learned reliability multiplier
uses four causal features: selected triangle–far-value agreement, query–value
agreement, valid far-modal coverage, and modal-content dispersion. It cannot
read the target word or target byte sequence. Fully masked far histories still
write exactly zero. The new scalar gain starts at zero and the source
predictions are bitwise identical to R7.37 before candidate-only warm start.

The added parameters form optimizer group 22. All old parameter identities,
groups, AdamW state and inherited optimizer engines remain untouched. The
single unified `source/fa1.py` entry point enables this layer by default;
the historical raw CLI retains an explicit `--triangle-reliability-r738` flag.
Loading a candidate marked `FA1_R7_38_CAUSAL_RELIABILITY_V1` reconstructs
the exact module sequence before strict model and optimizer state loading.

## Next diagnostic — quality must discriminate actual information

Reliability-off and reliability-on predictions are compared with the same
context on each DEV position. A second pair repeats the comparison after
replacing only eligible far-history *content* with a foreign-book donor;
receiver far-coverage is held fixed. The difference-in-differences effect is:

`(NLL_off_true - NLL_on_true) - (NLL_off_foreign - NLL_on_foreign)`.

This must improve word and conditional literal-byte NLL with positive
independent-book confidence intervals, alongside the existing hierarchy,
context, history, apex, writer, branch and byte gates. A positive training
gradient, an improvement on spelling alone, or a reduction in total loss
without independently useful far-history specificity is not enough.

At training ablation steps, `--r738-specificity-weight 0.15` adds a bounded
soft-margin penalty for this within-model effect. It is computed with the
correct and foreign writer histories at fixed recent context; no target is
used to form the query or memory address. `--r738-specificity-margin 0.003`
controls the margin. The model/optimizer are not replaced.

## Tests and reproduction status

- CPU source regression command:
  `PYTHONPATH=fa1r7.12/source python -m unittest discover -s fa1r7.12/source -p 'test_fa1_*.py' -q`
- Local integration: **85 tests passed**. Includes bitwise closed-gate
  predictions, Adam moment continuity, saved state roundtrip, strict
  repeat-attach rejection, masked-history invariance, bounded output,
  reliability-off ablation restoration, and word/byte gradient flow.
- These source tests **do not** establish held-out predictive improvement.
- GitHub Actions `fa1-r733-matched-far-history.yml` now compiles the new
  module and runs the new source suite when changed.

The next checkpoint-backed CPU round must start from the exact R7.37 parent
and recover its tokenizer tree and NaNROUTER book-disjoint split; it must
write an *unpromoted*, independently recoverable R7.38 successor at a different
path with all 22 optimizer groups and a parent SHA-256 record. Re-evaluate
multiple DEV seeds/positions and whole-book intervals; do not unseal TEST to
select a candidate. The accepted model is still protected at step 14,416.

**Metric scope:** conditional spelling BPB excludes end-of-word and is not
continuous raw-text bits/byte; inherited trunk top-k is not full-mixture word
top-k. Report separate word NLL/top-k, first-byte top-1/top-5, conditional
spelling top-1/top-5/BPB, and raw-text BPB only if a true causal raw-text
evaluation implementation is available.
