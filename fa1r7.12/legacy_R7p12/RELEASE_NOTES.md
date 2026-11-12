# PAL R7.7 release notes — promoted step 14,416

## Lineage and preservation

Recovered and SHA-256-verified the R7.6 step-14,000 archive (`b48b405018024d5c3883351597b2ba8db4a50cd14df2ad1638fe584f6740f419`). Migrated that *single model* into R7.7 with a low-rank learned novel-key associative path, adding only its new trainable parameters to the original fifth retrieval optimizer. Completed 416 additional joint CPU updates without resetting historical model parameters, optimizer slots, or RNG. Accepted checkpoint SHA-256: `54e835b00642013d4f1bf486a8150eb1ff06c547565771bd21933e143d050f2d`.

## Completed experiments

1. **Architecture migration** and novelty-key causal regression tests, 12 of 12 PASS.
2. **Learned association** training, step 14,000→14,416, using original historical training corpus.
3. **Matched ablations** on 4,096 historical development and 4,096 historical test positions, with 1,024-position shard logs.
4. **Generation smoke** from the actual model mixture; exploratory text remains fragmentary.
5. **Whole-book held-out evaluation pipeline** implementing file-hash split, corpus-overlap screening, fixed tokenizer, zero updates and per-book 8,192-token warmup. Verified engineering pipeline on synthetic fixtures only. Genuine NaNROUTER release archive unavailable inside this container.
6. **Rank-64 recall probe** showing no correct-state advantage. Optional 128-step outer-memory auxiliary training *rejected* because state-shuffle advantage worsened.
7. **Structural proof** that order information within each 64-token old-history block is erased by the existing initializer's mean. Planned gated order-sensitive migration described in research note; not trained here.

## Final historical audit

| Metric | Development | Test |
|---|---:|---:|
| Final NLL (active) | 5.138900 | 5.774114 |
| New semantic-only incremental gain | 0.000464 | 0.003880 |
| Pair-only incremental gain with semantic disabled | 0.008411 | 0.008198 |
| All retrieval gain | 0.092142 | 0.114659 |
| Outer matrix incremental gain | 0.000103 | 0.000107 |

The small gains are *historical*, not independent-book or statistically independent test performance. No fluent/free-generation capability is established.

## Rejected checkpoint

A temporary step-14,544 experimental continuation used the opt-in outer-memory auxiliary loss. Although its probe CE improved, it preferred *shuffled* older memory even more strongly; it was not promoted. The single active checkpoint in this release remains step 14,416. The negative probe and training logs are retained; rejected weights are excluded.

## R7.8 supplemental research — November 12, 2026

**Not a promoted checkpoint:** the rank-64 order-sensitive writer and directed historical-event extension were built and tested; 496 exploratory CPU updates reached step 14,912. Fixed 512-position development NLL improved by 0.003189, but fixed historical test NLL worsened by 0.002424 versus accepted R7.7 step 14,416. Correct historical memory did not consistently beat unrelated memory on test. The active checkpoint therefore remains R7.7, step 14,416. This archive includes the new code, training/evaluation logs and reproducibility report, but **no rejected candidate checkpoint**. See `research/R7p8_ORDER_SENSITIVE_MEMORY_EXPERIMENT.md`.

## R7.9 research continuation (non-promoted)

- Two separately evaluated continuations from the **same accepted** R7.7 14,416 state: +512 original-architecture updates to candidate 14,928 and +128 norm-calibrated, order-sensitive single-rank64-matrix updates to candidate 14,544. Both worsened development and test NLL on matched 512-position historical tests; **neither was accepted**.
- Measured old memory signal amplitude: effective read norm ~0.003–0.004 versus NOW ~2.5–3.0 in two illustrative R7.8 candidate contexts.
- Added `r79_normalized_memory.py`, `r79_compare.py`, `r79_eval_candidate.py`, `r79_rare_recall_probe.py`, `test_r79.py`; 30 inherited and new checks PASS.
- Identifiable unique-key, 32-choice old-event recall test found the correct physical state underperformed unrelated state on historical test for both original and calibrated candidates. No claim of independent book testing or generalization.
- Still exactly one accepted checkpoint in ZIP (step 14,416), five historical optimizers, no candidate `.pt` weights, and all necessary reproduction sources/logs. See `research/R7p9_NORMALIZED_READER_AND_CAUSAL_RECALL.md`.

## R7.11 selective directed recall: auxiliary-only experimental adapter

- One accepted full 14,416-step checkpoint unchanged. One 173,477-byte adapter overlay stores development-selected 80-step writer candidate on top of it; no second full `.pt` model and no optimizer reset.
- New 768-old-word directed 64×64 memory computes centered key/value covariance and trains on 32-way frozen-embedding-mined historical hard negatives. 320 attempted CPU auxiliary updates; chosen 80 due best development CE.
- On historical 512-query test, new reader correct-memory CE 3.3483, 16.80% top1, correct-vs-unrelated CE +0.5541, and correct-vs-reversed CE +1.1670. Alternate query seeds repeat direction. Not sealed independent books.
- Nonzero blend worsened 128-position dev next-word NLL slightly; **no promotion** or claim of better language modeling. Read research note for limitations and commands.

## R7.12 extension — streaming completed, memory-specific LM gate rejected

The R7.11 trained directed writer is now usable causally in a stream with a nonzero gate. A 768-token older-history ring and one persistent rank-64 matrix replace the former unsupported nonzero streaming mode. 64-position two-stream recurrence test passes exactly. A sparse directed-value LM probability-mixture experiment initially improved historical NLL, but a 512-query dev wrong-memory control outperformed correct memory: **do not promote**. A contrastive reliability-gate retrial also failed. A horizon sweep found the frozen writer does not benefit from simply increasing history to 6144 tokens. Parent checkpoint and five optimizer histories unchanged. Two compact overlays are included for reproducibility: inherited R7.11 directed writer and unpromoted R7.12 LM-gate calibration. Read the R7.12 research note before interpreting the results.
