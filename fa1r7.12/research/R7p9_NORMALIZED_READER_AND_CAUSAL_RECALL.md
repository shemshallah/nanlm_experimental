# PAL R7.9 — normalized mirror read + identifiable-cue recall (CPU)

## Release decision
**No promotion.** Accepted remains the single R7.7 step-14,416 checkpoint `checkpoint/pal_active.pt`, SHA-256 `54e835b00642013d4f1bf486a8150eb1ff06c547565771bd21933e143d050f2d`. Five optimizer histories and the rank-64 physical S+/S− matrix are retained. R7.8 unaccepted weights are not part of the new lineage.

This is a cumulative reproducibility release of an R7.9 investigation, not a replacement model. All training comparisons use the **existing historically consulted PAL train/development/test corpora**; no independent NaNROUTER book asset was obtained.

## Measured scale mismatch in the R7.8 trial

A direct read probe of local, unpromoted R7.8 step-14,912 weights on two historical test contexts found physical state norms ~0.50 and memory read norm ~0.14–0.21, compared with original NOW state norm ~2.5–3.0. With `outer_gain` ~0.01988, effective memory residual norms were **0.0028 and 0.0042**. This is roughly 0.1–0.2% of NOW amplitude. The R7.8 writer could distinguish order without influencing prediction much.

The probe uses the **unpromoted** R7.8 candidate saved in this session's transient working directory, excluded from the release. These are two illustrative examples, not corpus-wide medians or a general information-theoretic bound.

## Two CPU trials, both starting from accepted 14,416; neither accepted

### Trial A: continued original architecture

- +512 joint updates, separate experimental candidate step **14,928**.
- All five optimizer histories and original model parameters carried forward without reinitialization.
- Same `r77_semantic.py` training loss/optimizer and RNG; no new architecture or corpus.
- Matched historically consulted 512-position positions on both splits:

| Split | Accepted step 14,416 | Continued trial step 14,928 | NLL difference (trial − accepted) |
|---|---:|---:|---:|
| Development | **4.953608256** | 4.953817021 | +0.000208765 |
| Test | **5.832248628** | 5.832500204 | +0.000251576 |

The continuation failed the historical gate. Training loss itself is noisy and should not be used for checkpoint selection.

### Trial B: zero-migrated calibrated read

`source/r79_normalized_memory.py` builds on accepted R7.7, upgrading the **same** physical rank64 matrix with R7.8's zero-gated chronological and directed pair writers. It changes the downstream read into

`atanh(now_new) = atanh(now_base) + outer_gain·read + tanh(normal_gain)·RMS(now)·clamp(read/RMS(read), -3, 3)`.

The calibrating gate `normal_gain` starts **exactly at zero**, so migration reproduces the accepted checkpoint predictions and original optimizer tensors exactly. The parameter is added to the *existing fourth optimizer* with a new parameter group, retaining exactly five optimizer histories. Only one rank64 state matrix is used; no independent decoder/model or separate matrix is introduced.

- +128 joint CPU updates, experimental candidate step **14,544**.
- Matched historically consulted 512-position measurements:

| Split | Accepted step 14,416 | Calibrated reader trial step 14,544 | NLL difference (trial − accepted) |
|---|---:|---:|---:|
| Development | **4.953608256** | 4.955024190 | +0.001415934 |
| Test | **5.832248628** | 5.836842153 | +0.004593525 |

The normal gain reached ~+0.00121; directed pair gain ~−0.07225; order gain ~0.000498. The better residual scaling gave a learnable signal, **but did not improve next-word prediction**. The 128-position test-split correct-state versus unrelated-state advantage was −0.00001134 nats/token and normalized-read-off advantage −0.00009494 nats/token.

## Rare unique-key historical recall probe

`source/r79_rare_recall_probe.py` picks a past word that occurs **once** in the older 6,144-token history, restricted to the most recent 512 of those old tokens; its true immediate following old word is a positive. Exactly 31 negative word types are drawn from that *same causal history*. The query, candidate set, and target are identical when comparing the correct memory with an unrelated batch-swapped memory. We score each candidate by normalized existing embedding direction and the same memory query/read projection. These are historic split positions, not unseen books. Query-level bootstrap intervals do **not** imply independent book-level confidence.

| Model/split | Correct recall CE | Unrelated recall CE | Benefit (unrelated − correct) | 95% query bootstrap interval |
|---|---:|---:|---:|---:|
| Accepted / development | 4.013765 | 4.019366 | +0.005601 | [+0.00015, +0.01135] |
| Accepted / test | 4.010879 | 4.005284 | **−0.005595** | [−0.01354, +0.00235] |
| Calibrated trial / development | 3.926746 | 3.933098 | +0.006352 | [+0.00076, +0.01226] |
| Calibrated trial / test | 3.931575 | 3.924841 | **−0.006734** | [−0.01505, +0.00160] |

Each row tests 128 identifiable queries, all within the historical corpus. Both trial and accepted models fail the held-out test's correct-memory advantage diagnostic; neither is evidence of generalized associative recall. The mean CE also exceeds chance-uniform `ln(32) ≈ 3.466`, indicating the query-to-value read is miscalibrated for this task. Within-block reversal gives identical values for accepted R7.7 as expected and changes the calibrated trial's values, proving information preservation but not useful recall.

## What was tested

- Original R7.7 inherited regression: **12/12** checks.
- R7.8 ordered writer: **6/6** checks.
- R7.8 directed adjacency writer: **5/5** checks.
- R7.9 migration: **7/7** checks (exact original tensors and Adam moments, original logits unchanged, one physical matrix, causal order sensitivity when enabled, and nonzero normalized gain gradient).
- Total **30/30** structural/regression tests; **no positive validation result**.

## Scientific diagnosis and next objective

The unresolved problem is not just residual amplitude. A long-term rank64 state must retain **key-specific, order-sensitive information that a query can retrieve**. Averaging 64-word chunks destroys order; an unselective 6,144-pair matrix superposition can retain order but suffers capacity interference and may respond worse to the correct history than to unrelated history. The published R7.9 calibrated read corrects gain mismatch without making associations recoverable.

Next experiment (not executed here): construct a selective, causally weighted directed-history writer with controlled event count in the *same* matrix and train identifiable key→next-value recall. Pre-register the correct-vs-unrelated and correct-vs-reversed-memory tests. Only after demonstrating improvement on held-out historical books, add this read into the NOW prediction residual and assess word NLL on verified independent Gutenberg/NaNROUTER books. Do not increase rank/context size or claim frontier performance until those gates pass.

## Reproduction

```sh
cd extracted_archive
export PYTHONPATH=source OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
python source/test_r77.py --root .
python source/test_r78.py
python source/test_r78_bigram.py
python source/test_r79.py
# Experimental scripts write candidate weights OUTSIDE the archive's active checkpoint.
python source/r77_semantic.py --root . --resume checkpoint/pal_active.pt \
  --output /tmp/pal_r79_standard_trial.pt --steps 512 --streams 2 --seq-len 2 \
  --eval 0 --threads 2 --checkpoint-every 256
python source/r79_normalized_memory.py --root . --resume checkpoint/pal_active.pt \
  --output /tmp/pal_r79_normalized_trial.pt --steps 128 --streams 2 --seq-len 2
python source/r79_rare_recall_probe.py --root . --candidate /tmp/pal_r79_normalized_trial.pt \
  --n 128 --output /tmp/pal_r79_rare_recall.json
```

A new-book evaluation remains blocked by unavailable GitHub release binary assets in this runtime. `shemshallah/NaNROUTER/corpora/README.md` confirms the large corpora are distributed as GitHub Releases rather than normal repository files. The original source archives, checkpoint, tokenizer, trained matrix, historical data and test programs are included.
