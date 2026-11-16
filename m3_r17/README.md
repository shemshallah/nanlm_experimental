# M3 R17 — Precommitted independent-book TEST

R16's six DEV books showed a small **positive** spatial-wave gain over its oscillator carrier: **+0.00231924 BPB**, with 5/6 books improving and a descriptive whole-book 95% bootstrap interval of **[+0.00036013, +0.00474814]**. This motivated a *single* independent TEST, not further tuning.

## Freeze before TEST

- Exact coefficients from [successful R16 GitHub Actions run #43806939769](https://github.com/shemshallah/nanlm_experimental/actions/runs/43806939769) are hardcoded in `locked_test.py`. No optimizer or fit method is called on TEST.
- Spatial wave selected using R16 DEV; scale **1.0** selected using R16 TRAIN inner holdout. R17 does not change either.
- Eight original TRAIN documents determine the unigram prior; six original DEV documents are **fingerprint-verified only**; six new TEST documents are scored once. Document selection seed **160016**, corpus batch 0004, 4096-byte UTF-8-aligned prefixes, 1024-byte warmup. The original 14 SHA-256 hashes must match before scoring TEST.
- M3's exact 256-head + 32,640 mirror-pair wave machinery stays intact. FA1, FA2, Terminal-RSI and the historical M3 R12 checkpoint are unaffected.

## Predeclared interpretation

Primary statistic: TEST whole-book bits/byte reduction vs the *same* frozen oscillator, with per-book signs and descriptive whole-book bootstrap 95% interval. Also report memory-only, top-1/top-5, nonspace BPB, UTF-8 bits/character, and throughput. A positive interval lower bound strengthens evidence; non-positive gain or an interval spanning zero is **not confirmation**. Because R16 selected spatial from DEV and TEST has only six books, even a success warrants broader-corpus replication before promotion or generalization claims.

**Run**: GitHub Actions workflow `m3-r17-locked-wave-test.yml` on `master`, which publishes a JSON artifact. The workflow tests its software before reading the independent books. Do not retrain on TEST or silently revise frozen coefficients after seeing its outcomes.
