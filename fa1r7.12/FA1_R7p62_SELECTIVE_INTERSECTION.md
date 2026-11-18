# FA1 R7.62 — Selective two-rail geometric intersection

**Status:** DEV-only CPU candidate. No automatic promotion. Independent TEST sealed.

## Single-lineage migration

R7.62 continues the complete *unpromoted* R7.60 checkpoint with all 34 AdamW
optimizer groups. R7.61 was evaluated but failed the strict lexical and older
history gates, so its unpromoted parameters are not adopted. The R7.62
architecture appends exactly one new optimizer group; all inherited weights and
moments remain bitwise unchanged. The default accepted model is not replaced.

## New geometric computation

At each causal `(older_word, recent_word)` query, two 256-point abstraction
rails define three sparse conditional successor distributions:

1. Head-0 marginal: `(recent_word, head0(older_word))`.
2. Head-1 marginal: `(recent_word, head1(older_word))`.
3. Oriented intersection: `(recent_word, head0(older_word), head1(older_word))`.

All tables are built from TRAIN books only; no DEV/TEST outcomes enter memory.
Each rail is independently smoothed toward the exact recent-word lexical
successor distribution, using its own trainable pseudocount. A trainable
four-way softmax, conditioned only on pre-target support, chooses head0,
head1, their intersection, or a lexical-only null expert. The resulting
distribution passes through the inherited exact-word hierarchy and a bounded
residual gate. Zero-gate attachment is exactly function preserving.

## Required causal and statistical gates

- 270 unique TRAIN documents, 15 independent DEV documents; TEST remains sealed.
- One-thousand-update CPU pilot, inherited optimizer moments verified unchanged.
- Compare same DEV positions against R7.60 parent, lexical-matched control,
  wrong older-word history, and wrong geometric values.
- Require positive per-book bootstrap lower CI95 for all four gains before
  considering +9,000 updates or production promotion.
- Report real NLL in nats/word. Byte BPB and byte top-5 are not comparable to
  these word-level scores and are not estimated here.

The expected causal hypothesis is that selecting a reliable rail or
intersection improves prediction where a fixed pooled average does not.
The hypothesis is not a claim of superiority until the gate passes.
