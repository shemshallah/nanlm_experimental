# M3 R18 — Frozen spatial-wave replication, 32 more books

R17 confirmed **+0.0034971906 BPB** relative to the matched oscillator over six
previously unseen documents (18,432 scored positions, 6/6 positive). However,
its **non-space** BPB worsened from 4.4676841 to 4.4683181. That calls for
an untouched expansion, not tuning on those TEST outcomes.

## Predeclared R18 protocol

1. Continue the **independent M3** experiment. Do not merge FA1, FA2, or
   Terminal-RSI into this architecture, and do not claim an R12 checkpoint.
2. Use the NaNROUTER `corpora-short-stories-0004` release with verified
   SHA-256, seed `160016`, 4,096-byte UTF-8-aligned prefixes, 1,024-byte warmup.
3. Preserve indices 0–7 (R16 TRAIN) for the unigram prior only, indices 8–13
   (R16 DEV) and 14–19 (R17 TEST) for fingerprint verification only, and score
   **indices 20–51** (32 new documents) exactly once. Every selected byte
   prefix is SHA-256-locked in `frozen_split.json` before GitHub evaluation.
4. Reuse the exact R16/R17 frozen oscillator and graph-residual coefficients
   through `m3_r17.locked_test.frozen_predictions`. No fitting, reselection,
   normalization changes, selective fallback, or document-conditioned gate.
5. Stream individual books and aggregate by scored byte, not equal-weighted
   book; calculate whole-book bootstrap intervals with whole books as units.
6. Report overall BPB/BPC, top-1/5, non-space BPB, throughput, per-book signs,
   and **space vs non-space** BPB contributions separately. Bootstrap intervals
   are descriptive; the same release does not establish out-of-domain transfer.

This test is an *evidence expansion*, not a trainable successor. Nothing is
promoted automatically. The next gate, if this succeeds, is a capacity-matched
non-geometric recurrent control and independent-corpus long-context evaluation.

Run on GitHub `master`: `.github/workflows/m3-r18-frozen-replication.yml`.
