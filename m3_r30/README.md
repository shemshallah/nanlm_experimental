# M3 R30 — selective lexical trust and honest history comparison

R29 is the **unified default**. R26 remains the last independently certified
model. R30 is a zero-initialized, experimental continuation within the **same
M3 lineage**, not a replacement network or a new Git branch.

R29's 22 Adam groups and coefficients are verified against complete checkpoint
SHA-256 `02c340e82ec77ae2c6e1d20ddd808ca4f63e3c242fe6fc94ff8f2235d8ab9386`.
Three new scalars learn corrections to its existing lexical logit feedback:

- word-prefix spelling evidence;
- previous-word-conditioned evidence;
- support-conditioned previous-word evidence.

All three initialize at zero, exactly reproducing R29 logits. Training adjusts
only three new Adam groups and carries all 22 frozen groups (including their
first and second moments and independent step ages) into the new checkpoint.

## Fresh split, sealed TEST

Same SHA-verified NaNROUTER Release batch 0004 and fixed TRAIN-only unigram
prior. Original R26 TEST 276–299 and R29 unopened TEST 332–355 are never used.

| Set | Indices | Use |
|---|---|---|
| R30 FIT | 356–371 | 768 CPU Adam mini-batch updates |
| R30 GATE | 372–375 | candidate acceptance check |
| R30 DEV | 376–387 | independent-book model selection |
| R30 TEST | 388–411 | **SEALED, not read** |

Unlike R29's original unrelated-history control (which preloaded all of an
unrelated book), the *matched* control observes exactly one unrelated byte for
each observed real byte, strictly after scoring. A separate `unrelated_full`
arm preserves the harder full-unrelated-book diagnostic without conflating
its additional context with causal retrieval superiority.

Reported arms: promoted R29, R30, wrong-successor R30, matched unrelated,
full unrelated. Metrics: next-byte BPB/BPC, nonspace BPB, top-1/top-5,
per-book deltas, and descriptive whole-book bootstrap uncertainty. Candidate
must beat R29 and the wrong/matched controls without accuracy regressions;
full unrelated is reported separately. Promotion requires further frozen
independent TEST, not merely a DEV win.
