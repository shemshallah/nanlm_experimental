# FA1 R7.63 — Causal temporal cross-rail geometry

**Status:** experimental CPU candidate; no promotion or TEST access until
book-disjoint superiority is demonstrated.

## Unified inheritance

Start from the *complete* R7.62 35-optimizer-group unpromoted checkpoint.
Keep inherited parameters and AdamW moments frozen. Append a single 36th
optimizer group for four new tensors. The R7.62 checkpoint is immutable and
has a full recovery archive; no parallel neural predictor is created.

## Geometric memory

R7.62's three head-memory addresses depend only on word `t-2` and exact `t-1`.
R7.63 consumes another causal word `t-3`, composing two distinct oriented
crossings of the existing 256-point head rails:

- `S+ = (recent_word, head0(t-3), head1(t-2))`
- `S- = (recent_word, head1(t-3), head0(t-2))`

Every four-gram count and successor outcome is obtained only from the
original 270 TRAIN books, with explicit document-boundary exclusion.
Each rail obtains a separate learned lexical shrinkage and the selector
can choose S+, S-, or lexical-only null. The selected distribution is
composed with the inherited exact-bigram hierarchy. A zero-gain attachment
must exactly preserve R7.62, which remains in the single inference path.

## Evaluation protocol

Run 1,000 CPU updates first on 4,096 causal TRAIN contexts and 2,048
held-out DEV contexts spanning 15 disjoint books. Apply book-bootstrap CI95
to gains over R7.62 parent, matched lexical control, unrelated older-word
history, and corrupted geometric values. All gains must be positive with a
positive CI lower bound before proceeding to 9,000 more updates or opening
independent TEST. Preserve all optimizer groups and check exact reload.

This is a word-level NLL experiment; byte BPB and byte top-5 are not inferred.
