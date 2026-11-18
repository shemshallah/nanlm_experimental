# FA1 R7.64 — Multiresolution temporal Diamond

**Experimental status:** Development-only candidate. The accepted production
checkpoint and all independent TEST books remain protected. Promotion and
9,000 extra CPU updates require the *same* independently sampled book-level
superiority gates used in R7.63.

## Why R7.63's finer crossing stalled

An address composed of an exact recent lexical token and two 256-point
head coordinates has 65,536 conceptual subaddresses **per recent word**.
The NaNROUTER TRAIN sample has 6.9 million word positions; the distribution
of visits across such fine addresses is sparse. Even a true temporal
association may have too few observations to alter next-word probability.

## Single-model repair

R7.64 **adds** four lower-resolution TRAIN-only conditional distributions to
the same model; R7.63 and all earlier paths remain intact:

| Address | Resolution | Evidence |
| --- | --- | --- |
| S+ coarse | `(head0(t-3), head1(t-2))` | 256×256 |
| S- coarse | `(head1(t-3), head0(t-2))` | 256×256 |
| S+ lexical | `(word(t-1), head0(t-3))` | V×256 |
| S- lexical | `(word(t-1), head1(t-2))` | V×256 |

The original finer R7.63 S+/S- banks remain active. A trainable two-layer
selector sees four counts, inherited exact and lexical supports, and head
agreement. It chooses among four Dirichlet-shrunk geometric distributions
and a lexical-only null, with a separately learned magnitude gate.
The resulting mixture remains normalized across successor vocabulary,
and its selector is independent of the supervised successor target.

The R7.64 state has six new trainable tensors in **one** new AdamW group.
The first 36 groups (all learned weights and optimizer moments) must remain
bitwise identical. Initial gate zero preserves the R7.63 distribution
exactly, and the inherited path remains usable at all times.

## Causal protocol and constraints

TRAIN banks are built from original NaNROUTER Release
`corpora-short-stories-0004`, never DEV or TEST. Book-boundary-crossing
four-grams are excluded. Supervised TRAIN counts use leave-one-out evaluation;
counterfactual value and history controls do not deduct nonexistent events.

First experiment: 1,000 CPU updates over book-balanced TRAIN positions,
with 15-book DEV evaluation. Required evidence:

1. Matched DEV NLL gain over R7.63, lower bootstrap CI95 strictly positive.
2. R7.64 gain over matched lexical-only alternative, lower CI95 positive.
3. Correct older-history advantage over corrupted older words, lower CI95 positive.
4. Correct geometric-value advantage over corrupted successor values, lower CI95 positive.

Only if every independent-book confidence bound passes is a +9,000
continuation eligible for consideration. TEST must remain sealed until the
architecture and training decision are frozen.

This is word-level NLL, not measured byte BPB or full vocabulary top-5.

## User-authorized DEV promotion

R7.64 completed the 1,000-update CPU DEV gate with positive 15-book
bootstrap lower bounds against all four competitors. The promotion script
requires an additional fresh-position DEV replication before it marks the
same trained 37-group checkpoint as the FA1 default.

**Reproducibility finding, 2026-11-18:** The original 1,000-update result
reproduced on the same deterministic samples and passed all four gates.
However, the separate 4,096-position, 15-book DEV sample produced a
positive mean hierarchy advantage (+0.00074046 nats/word) and positive
geometry-versus-lexical advantage (+0.00079175 nats/word), **but both
book-bootstrap lower bounds were negative** (-0.00051495 and -0.00035131).
The candidate therefore failed this stronger replication and must **not**
be promoted on that evidence. Historical positive results are retained;
independent TEST remains sealed. A same-architecture continuation resumes
the full R7.64 state and sixth-tensor AdamW history while freezing all
36 inherited parameter groups. Candidate selection uses original DEV
samples; any later replication must declare its own sample/seed and
acknowledge that DEV has been consulted repeatedly.

The default entrypoint is `source/fa1_default.py:load_default` and its
immutable checkpoint filename is `FA1_R764_PROMOTED_DEV_GATE.pt`. This
changes FA1's default route only. R7.63 and older explicit checkpoint
paths still load unchanged. The promoted metadata states
`DEV_verified_default_TEST_unopened`; this is not a claim of independent
TEST or frontier superiority.
