# NaNCM v176 — Symbiotic tri-coding, signed-concept word top-five + speculative byte verification

**Justin Howard-Stanley · Independent Researcher · Santa Cruz, California, USA.** 2026-11-23.

## Hybrid architecture — ONE checkpoint lineage, no competing master branch

The source-basis is the uploaded `v133_iterate.zip` (`v133/nancm.py`, `v131/nancm.py` and metrics). v133 is **source only**: no trained v133 sidecar was present. Original v126/v133 5MB online-training word-top-five numbers (v126 75.4%; v133 online 72.9%, frozen 53.6%) do **not** certify this hybrid or transfer to independent books. All third-coding BPE merges and root/letter histograms were retrained on the **396-book NaNROUTER Release0003 TRAIN** split; no v133 weights were imported.

1. **128+128 point signed S+/S− cloud → 64+64D recurrent v175 NOW state.** A frozen inherited 12,000-way whole-word expert and causal repeated-prefix copy support phrase history. Preserve the v175 checkpoint, four-group AdamW optimizer (19 states), the v174 dual-rail optimizer, and v173 weights.
2. **Trainable 14-feature counterfactual trust gate** predicts a bounded convex mixture of v175 copy-enabled and copy-disabled distributions, conditioned only on past-word repetition and model confidences; 24,000 TRAIN examples, 55-book VAL select step **20**. The new gate is a separate fifth optimizer lineage, not destructive parent continuation.
3. **First-letter link** from TRAIN-only causal unigram/two-word statistics (Dirichlet smoothing prior 24), calibrated on VAL. Exact normalized 12k word distribution gets a first-letter likelihood ratio with power **0.50**.
4. **v133-style BPE first-token link** trained 1,600 byte merges exclusively from the 95,040 TRAIN words, with 1,856-token dictionary. Its causal first-segment distribution, smoothed prior 24, contributes a normalized conditional likelihood ratio at **0.35**, selected on VAL.
5. **Third code** uses 5 full-word drafts + learned BPE prefix trie + **positive-probability byte escape**. `wf176_hybrid_open_full` loads inherited native byte model + trained sidecars. The 256-way verifier P is authoritative for *distribution-preserving* speculative sampling: draw b~Q, accept min(1,P(b)/Q(b)), otherwise draw from normalized (P−Q)+ residual. On falling off the trie Q becomes P. The C++ 256-way byte oracle remains unchanged.
6. **Separate lossless range-code mode:** arithmetic coder quantizes time-varying 256-byte P distributions into strictly positive integer frequencies and round trips arbitrary bytes. This is a codec; adopting a new hybrid probability P′ for compression must be scored separately from speculative acceleration, with true contiguous text and source boundary handling.

## Matched full-12k in-vocabulary word benchmarks (no OOV claim)

| Cohort | Metric | Frozen v175 | Hybrid v176 |
|---|---|---:|---:|
| 55 VAL books (n=13200) | Word NLL ↓ | 5.939311 | **5.817160** |
| 55 VAL books | Strict top-1 | 1573 | **1687** |
| 55 VAL books | Strict top-5 | 3596 | **3826** |
| 49 previously inspected TEST books (n=11760) | Word NLL ↓ | 6.004215 | **5.882440** |
| 49 reused TEST books | Strict top-1 | 1364 | **1483** |
| 49 reused TEST books | Strict top-5 | 3123 | **3354** |

The 49-book TEST cohort was exposed during v175 development, and inherited pretraining may overlap this corpus: **diagnostic improvement, not pristine certification**. Current hybrid reused-TEST strict top-1 is 12.61%, very far from an 80% target. The top-five result reflects the improved 12k specialist; v133's historical 75% online top-five is NOT being claimed.

## Native parity and code tests

- 1,000 complete 12,000-way probability vectors native vs PyTorch, **8.35e-7 max absolute difference**, identical top-1/top-5; 1,000 **bit-exact** v175 12k native rollback vectors when hybrid off.
- 144 newly checked, complete 256-way inherited byte rollback distributions **max difference 0**. Older full v175 16,018-byte DEV and reused-TEST replay is inherited and unchanged; full 16k v176 fresh replay not claimed.
- 32-context top-five native word draft and beam prefix byte re-scoring pass. 160-context sampler accepted ~2.57 bytes/draft, emitted ~3.13 before stop/rejection. Acceptance tests exhaust 15×256 synthetic cases and 90,000 empirical target samples. **No wall-clock inference speedup** measured: verifier is sequential.
- Exact lossless 32-bit arithmetic coding round-trip includes 1,200 random bytes and adaptive distributions.
- Native first-letter loss, *reconstructed word-only histories*: 280 VAL **4.056697→3.975556 bits**, 220 reused-TEST **4.127049→4.066600 bits**. This is NOT natural full-text BPB or BPE BPB.

## Build and reproduce from the single recovery ZIP

```
make -B v176
make v176-source-smoke
OPENBLAS_NUM_THREADS=1 python experiments/nancm/v176/test_crosslevel176.py --n 1000 --bpe
OPENBLAS_NUM_THREADS=1 python experiments/nancm/v176/test_byte_parity176.py --n 320
OPENBLAS_NUM_THREADS=1 python experiments/nancm/v176/eval_bpe_crosslevel176.py
OPENBLAS_NUM_THREADS=1 python experiments/nancm/v176/bench_speculative176.py --n 160
```

The cumulative recovery ZIP includes all 861 original v175 entries (the original Makefile is saved as `ancestry/v175_Makefile`), all new native source and trained sidecars, source v133 upload, complete optimizer-bearing parent checkpoints, training/validation/test fixtures, research and claim ledger, checksum manifest and SHA256SUMS. GitHub master carries source only. The user-directed **one ZIP in Library per NaNCM lineage** applies after the new archive passes upload-back integrity.

## Promotion policy

**Promote v176 as a measured improved 12k standalone hybrid word specialist, not as active open-vocabulary whole-word or byte decoder.** Keep v167 active native open-word path and v170 active byte path unchanged, with opt-in third-code proposals. To move beyond research, evaluate untouched author/book-disjoint targets, natural contiguous text, net byte BPB and actual wall-clock parallel batching/reuse, and compare to unrelated histories.