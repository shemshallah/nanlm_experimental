# FA1 — in-place apex-to-language feedback

This is a modification of the **same FA1 R7 Diamond**, not R7.35 as a
replacement architecture. It is applied through `source/fa1.py` on `master`.
Historical model parameters, native pair/triangle addresses, S+/S− channels,
tetrahedral specialists, and original decoder remain attached.

## Causal feedback path

After the existing 256-point → 32,640 pair → sparse triangle → 256-specialist
apex → pair → point descent, a small cross-attention reader considers:

1. The descended 32-dimensional point message.
2. The selected apex specialist's 32-dimensional state.
3. Their directional interaction, point × tanh(apex).

The previous-context query attends to these three features. The resulting
32D state has separate projections into the inherited 96D word correction
and the 32D autoregressive spelling condition. The contribution is gated by
`tanh(apex_feedback_gain)` and by the product of descending-point and apex
route strengths, plus the preexisting calibrated word/byte trust.

This carries geometry back into both language heads while keeping the
original byte/subword/token/word/sentence context loops, sparse geometric
attention, multi-horizon far-history curriculum and matched donor controls.
The attention does not inspect a target word or future byte: all three memory
tokens originate from the context-only Diamond forward pass.

## Migration and training protection

- The new gain initializes to exactly **zero**: attached and unattached
  logits are bitwise equal at initialization, even if existing gates learned
  nonzero values.
- The new attention module and scalar gain append one AdamW group to the
  existing unified optimizer (17 → 18 in the full calibrated lineage).
- Previously trained parameter identities, old Adam moments, legacy
  accumulator and accepted step-14,416 ancestor are retained.
- The same `fa1_r727_checkpoint` serializer records the new feature within
  the existing FA1 lineage using `FA1_R7_UNIFIED_APEX_FEEDBACK_V1`, so the
  loader attaches exactly the needed modules **before** strict state restore.
- A candidate is always unpromoted. The canonical CLI runs DEV only and
  guards source checkpoint/report/candidate paths.

The full feedback ablation additionally requires positive word, branch,
and literal-byte gains, and strictly positive book-bootstrap lower bounds for
word and byte gains. The existing geometry, matched-history, matched far
controls, conditional byte top-1/top-5, and word NLL gates still apply.

## Verification and next scientific test

```bash
PYTHONPATH=fa1r7.12/source python -m unittest discover \
  -s fa1r7.12/source -p 'test_fa1_*.py' -q
```

Source tests require exact zero-gate parity, nonzero gain gradient, causal
byte masking, route-dependent suppression, optimizer-moment continuity and
failure on negative feedback contribution. This does **not** demonstrate
a real-corpus language-model improvement. Next, load the SHA-256-verified full
FA1 checkpoint, continue only on book-disjoint NaNROUTER TRAIN, compare a
frozen DEV baseline and matched-control ablations, and open sealed TEST only
after strict DEV success. Report word and byte top-1/top-5, word NLL,
conditional spelling BPB and actual literal-byte counts. Raw contiguous-text
BPB is a distinct metric and remains unavailable.
