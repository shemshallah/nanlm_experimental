# PAL R7.7 claim and evidence ledger

| Claim | Evidence | Status |
|---|---|---|
| R7.6 parent recovered and verified | Archive CRC + original checkpoint SHA256 `b48b405018024d5c3883351597b2ba8db4a50cd14df2ad1638fe584f6740f419` | Verified |
| Current checkpoint is one continued 14,416-step model | `checkpoint/pal_active.pt`, SHA `54e835b00642013d4f1bf486a8150eb1ff06c547565771bd21933e143d050f2d` | Verified |
| All five optimizer histories persisted through migration/416 updates | checkpoint keys, optimizer state, `source/test_r77.py` | Verified |
| Learned non-exact key retrieval uses strictly older targets | `SoftAssociation.lookup`, causal checks in `test_r77.py` | Verified for implemented indexing |
| Novel path produces positive incremental historical test NLL benefit | `reports/R7p7_step14416_test_historical_4096.json`, +0.003880 nats | Historical only |
| Original one/two-word paths continue to help | Fixed evaluation ablations; +0.102580 exact-one-word, +0.008198 pair on test | Historical only |
| Outer rank-64 initializer retains within-64-token order | `CrossWindowMirror.initial`: mean over 64 tokens | **False: order provably lost** |
| Outer rank-64 state encodes usable historical associations | Correct-state advantage −0.004500; auxiliary trial −0.013189 nats | Not established, auxiliary rejected |
| New-book data pathway works | Synthetic three-file fixture audit; no training on fixture | Verified engineering smoke only |
| Real unseen Gutenberg-book generalization | Only NaNROUTER release metadata accessible; no original binary books downloaded | **Not evaluated** |
| Fluent language generation | Actual-mixture generation smoke is fragmentary | Not established |
| Competitive frontier model | No frontier-scale benchmark or comparable model | Not established |

## R7.8 order-aware memory research (not promoted)
- VERIFIED STRUCTURAL: Eight-bin temporal moment writer and directed key-to-next-value writer both inject information into the original single rank64 S+/S- matrix; disabling their gates reproduces original migration behavior. Six zero-gate/order tests and five pair-event tests pass; inherited 12 R7.7 regressions pass.
- VERIFIED TRAINING: Original 14,416 checkpoint and five optimizer histories were continuously migrated and trained up to 14,912 in exploratory candidate states; no optimizer reset. These rejected candidate weights are NOT included in this accepted-lineage release.
- FAILED PROMOTION: On exactly matched 512-position historical test, R7.8 NLL=5.834672540 > R7.7 NLL=5.832248628; correct-state historical-transition recall CE is also worse than unrelated state CE on test. The evidence does not establish useful order-aware generalization.
- NOT TESTED: NaNROUTER independent whole-book Gutenberg holdout, out-of-domain holdout, book-bootstrap confidence intervals, frontier comparisons.

## R7.9 assessment

**PROVED (implementation / structural tests):** R7.9 normal-gate migration is numerically prediction-identical at zero gain to accepted R7.7, preserves all pre-existing parameters and five optimizer states bit-for-bit, and a nonzero normalized gain affects prediction and receives gradients. Existing rank64 matrix remains unique. All 30 regression/structural checks pass.

**OBSERVED (historical split only):** Conventional +512-step continuation and new normalized-reader +128-step continuation both worsen matched 512-position development/test NLL. Unique-key older-history recall has unfavorable correct-over-unrelated-memory CE on test for accepted and new trial. These are not independently selected Gutenberg books and not frontier benchmarks.

**NOT ESTABLISHED:** Useful generalized chronological memory, robust held-out key/value recall, independent-new-book gains, scaling laws, fluency, or a frontier trajectory. Rejected candidate weights were not promoted, and no new accepted checkpoint was created.

## PAL R7.10 claim gate

- **Supported:** accepted step-14,416 checkpoint unchanged, full five optimizer
  and four RNG histories restored exactly; candidate atomic serialization,
  SHA sidecar rejection and accidental active-path overwrite guard tested.
- **Supported:** on 512 historically consulted causal recall queries per split,
  the accepted rank64 readout is worse than a uniform 32-way recall baseline
  under both random and embedding-similar competing-history negative sets.
- **Not supported:** useful generalizing rank64 associative retrieval,
  new-book performance, significant correct-memory advantage, an order-sensitive
  writer trained successfully, or further LM learning during R7.10.
- **Future:** selective event writer, hard-negative supervised recall and
  independent source-disjoint new-book gate before NLL promotion.

## PAL R7.11 directed associative memory continuation (not promoted)

- **SUPPORTED HISTORICAL EXPERIMENT:** Zero-gated 768-token directed key→next-value selective 64×64 matrix readout trained for 320 attempted CPU auxiliary updates. Development-selected update 80 achieved 20.31% top-1 in 32-way 512-query historical dev and 16.80% on the repeatedly consulted historical test (chance 3.125%). Correct vs unrelated memory advantage +0.58695 nats dev, +0.55413 test. Reversal inside 64-word blocks harms recall. Alternate query seeds yield positive directional recall evidence; neither is book-independent.
- **VERIFIED INTEGRITY:** 14,416 joint LM steps unchanged, accepted full checkpoint unchanged SHA `54e835...`, five inherited optimizer histories bitwise intact, 173,477-byte new-parameter overlay restores selected candidate all 79 tensors and 5 optimizers exactly, 10 R7.11 tests plus 45 inherited tests pass. 320 tried, 80 selected, optimizer snapshot matches selected weights.
- **FAILED LM PROMOTION GATE:** Nonzero memory blends all slightly worsened historical 128-position development NLL, so accepted gate remains zero and the new directed read is NOT used for normal language-model prediction or streaming. Correct streaming advance for a nonzero blend is not yet implemented.
- **NOT ESTABLISHED:** Book/author-disjoint generalization, valid book-bootstrap CI (old corpus lacks doc-start metadata), independent sealed test, long-range recall beyond 768 directed window, language fluency, or frontier scaling. Do not compare R7.10 original reader CE to R7.11 new reader CE as an equal-architecture baseline.

## PAL R7.12 frontier trajectory: evidence, not promotion

- VERIFIED: exact causal 64-position streaming directed-state recurrence; one persistent 64×64 rank-64 matrix plus 768-token ID ring; initial nonzero-gate equality with R7.11; zero gate reproduces inherited predictions; positive nonzero-gate LM gradient; accepted SHA unchanged. 2-stream CPU microbenchmark 1.759 ms per batch step for matrix read+advance ONLY.
- VERIFIED ENGINEERING: sparse 767-event directed-pair retrieval candidates and features are selected without looking at the current token target; mixture probability computed afterward; new fifth-optimizer parameter group, inherited five optimizer histories unchanged.
- HISTORICAL ONLY: 96-position development NLL gain +0.006589 from an auxiliary-trained gate. At 512-position new-seed development, correct gain +0.000261 but wrong-memory gain +0.007830; fails source-specificity gate. 512-position historical test correct-memory gain +0.006849; inconsistent and not book independent.
- NEGATIVE CONTROL: reliability-conditioned gate through 256 logged updates never satisfied dev positive-NLL and positive correct-vs-wrong requirements; no selected overlay. Terminal test not run due interruption; log retained.
- NEGATIVE SCALING: frozen writer horizon 768→6144 worsens 128-query dev CE 3.102794→3.673287; additional trained multiscale routing remains a hypothesis.
- NOT ESTABLISHED: improvement in promoted LM checkpoint, independent NaNROUTER books, book-bootstrap confidence, general long-range retrieval, frontier scaling, fluent generation.
