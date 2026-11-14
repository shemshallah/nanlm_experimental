# FA1 R7.36 — sparse causal readback, centered feedback and gradient audit

**Status: source-integrated candidate experiment, not a promoted model.** All
work remains on `master`, inside the same FA1 R7 network and unified optimizer.
The accepted step-14,416 checkpoint and the 15 sealed TEST books are untouched.
No new held-out language-model improvement is claimed by this source change.

## Faults isolated after R7.35

1. **Scale coupling.** The writer places far-token values at eight selected
   triangles, but its apex read multiplied the weighted sum by the full
   56-face count. This couples history amplitude to unrelated face occupancy.
   R7.36 computes an alternative read normalized by the attention mass on
   the *eight actual written addresses*. Its `history_read_mix` scalar starts
   at zero; the inherited R7.35 read is reproduced exactly at migration.
2. **Query-only feedback shortcut.** Apex-to-word/byte attention receives the
   causal query as a residual, so its output can contain query-only signal
   even if its specialist inputs carry no information. R7.36 also computes
   that attention's response to zero point/apex signals and can subtract it
   using `apex_feedback_center_gain`. This second scalar starts at zero,
   preserving the previous learned cross-attention and its outputs exactly.
3. **Misleading donor controls.** An unrelated book with every far token
   ineligible is an *empty-memory* control, not a wrong-*content* control.
   Counterfactual writer donors now come from a distinct book with at least
   one eligible far modal token. If no second eligible book exists, the probe
   fails explicitly. R7.36 minibatches enforce at least two eligible books.
4. **Book eligibility audit.** The diagnostic's `eligible_independent_books`
   now counts books with observed far-modal evidence rather than every book
   that supplied a target. Training reports separate finite/nonzero gradient
   norms for writer Q/K/V, sparse values, factor addresses, apex feedback,
   point/pair/triangle loops, byte decoder and both new controls. Non-finite
   gradients stop a candidate update before stepping the optimizer.

## Checkpoint and verification contract

- Migrate *in place* from the complete `FA1_R7_35_CAUSAL_HISTORY_VALUES_V1`
  checkpoint. Attach only two zero-initialized parameters in one additional
  AdamW group (R7.35's 19 groups become 20), retaining the inherited parameter
  objects, optimizer accumulators and learned gates.
- Save/reload using `FA1_R7_36_DEEP_WIRING_V1` under the existing R7.27
  serialization format; the accepted model cannot be overwritten.
- The canonical `source/fa1.py` entry point enables the optional R7.36
  modules. Historical explicit ablations retain their numbered CLIs.
- New metrics include `r736_sparse_*` and `r736_center_*` word, branch and
  conditional-byte gains with independent-book confidence bounds. Promotion
  also requires positive book-level gains for each, in addition to all
  previous word, history, apex and byte constraints.

Run:

```bash
PYTHONPATH=fa1r7.12/source python -m unittest discover \
  -s fa1r7.12/source -p 'test_fa1_*.py' -q
```

Unit tests cover exact old-output parity, preserved inherited moments, strict
state roundtrip, nonzero new gradients, apex-only value routing, correctly
normalized sparse writes, donor eligibility, minibatch repair and exception-
safe counterfactual control restoration.

## Current evidence and remaining limitation

The R7.35 fixed-96-position DEV benchmark remains the most recent measured
language result: word NLL 4.862853 → 4.862622; conditional spelling
bits/byte 5.001301 → 4.998825; conditional spelling byte top-1 5.45% →
5.72% and top-5 44.69% → 44.96%. These are **not** R7.36 measurements and
**not** raw continuous-text BPB. The R7.35 byte writer ablation and correct-
far-history advantage remain negative.

R7.36 needs the actual complete R7.35 candidate, at least 12 independent
long-history-eligible DEV books and a new controlled CPU continuation before
claiming predictive improvement. Its two gates should be evaluated separately
against their no-gain counterfactuals and the strict joint promotion gate.
