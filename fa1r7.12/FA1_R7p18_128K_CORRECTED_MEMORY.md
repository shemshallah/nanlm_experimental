# FA1 R7.18 — Future-safe 128K memory extends the frozen 32K reader

**Date:** 2026-11-12  
**Lineage:** FA1 only (PAL rank-64 temporal + hierarchical), never FA2/FA2.2/NaNLMind.  
**Status:** Historical CPU candidate selected, **NOT PROMOTED** to accepted unified model.  
**Accepted full parent:** step 14,416; SHA-256 `54e835b00642013d4f1bf486a8150eb1ff06c547565771bd21933e143d050f2d`, five original optimizer histories untouched.  
**Frozen prerequisite:** R7.16 step-75 8K/32K adapter SHA-256 `436ff87fae69ef1f44b53e2ffde92313da408342f65a1897c6eaff1876f9c09c`.  
**Selected R7.18 added 32K–128K adapter SHA-256:** `f4d9175ff202d9ec1ff3a06dcddfd3d262cea25f8441fa5dab972a313309aa9f`.

## Key correction: old 128K donor control could see future tokens

R7.17 used `np.roll` of positions within the same development split to generate supposedly unrelated histories. At a 128K memory horizon, on the 256-example historical development sample **33.2%** of donor histories belonged to **future** positions whose far-memory window could include the true target token. That was not a valid *causal control*; it could artificially improve unrelated-memory likelihood. The negative control advantage previously reported for R7.17 is therefore not reliable evidence that 128K memory itself is ineffective. R7.17 remains rejected, and no result is retroactively promoted.

R7.18 uses two future-safe counterfactuals instead:

1. **Shuffled old past:** reorder the words only inside the current example's existing older 32K–128K window; preserve its vocabulary frequencies while destroying chronological n-gram pairs. Neither the recent 32K nor any future token is touched.
2. **Training-split donor:** on development/test, rebuild all auxiliary n-gram routes from a distinct preexisting **training split**, never from a future position of the evaluation split. This is not definitive document independence because the old corpus has no book-level source manifest.

No target is used in indexing or in router features. Target-dependent likelihoods are used **only as training/evaluation losses**, not for candidate selection at inference.

## Architecture and freeze discipline

The original 96-dimensional inherited LM and its rank-64 memory are frozen and remain at the same checkpoint. The R7.16 8K/32K reader is loaded from its previously selected separate adapter and **also frozen**. R7.18 learns a tiny separate residual gate over four 1–4-gram query-conditioned histories in the disjoint age band **32,769–131,072 tokens ago**. It forms a bounded probability mixture over the R7.16 predictor and the four older-path values, with a no-memory option. New optimizer state belongs only to R7.18; all five inherited optimizer histories and R7.16 candidate remain immutable.

The extra history is sparse **exact-token n-gram retrieval**, *not* dense 128K attention, a semantic retriever, a full 128K autoregressive transformer, or evidence of frontier reasoning. The code currently rescans history per example and needs incremental indexing for CPU scaling.*

## Complete 1,024 / 512 / 512 CPU experiment

One run, 1,024 training positions, 512 development positions, 512 historical test positions, **240 optimizer updates**. Development-only selection chose update **48**. The reported test follows selection. Metrics are **nats per word**, not bits per byte, byte NLL, or subword perplexity.

| Historical evaluation | Original FA1 | Frozen R7.16 8K/32K | R7.18 + far 32K–128K |
|---|---:|---:|---:|
| Development NLL ↓ | 5.235429 | 5.146455 | **5.105531** |
| Historical test NLL ↓ | 5.945565 | 5.757913 | **5.516152** |

R7.18 provides an **incremental development gain of 0.040924 nats/word** over the frozen near reader, and an **incremental historical test gain of 0.241761 nats/word**. The historical test gain vs accepted parent is **0.429413 nats/word (7.22% relative)**. The larger test improvement than development is noteworthy but was not evaluated on document-independent books, and repeated historical split use limits statistical interpretation.

| Correct vs counterfactual auxiliary far history | Development advantage ↑ | Historical test advantage ↑ |
|---|---:|---:|
| Order-shuffled older past | +0.131381 | **+0.290978** |
| Separate training-split donor | +0.046193 | **+0.264486** |

The frozen near reader's NLL stays identical in each far-memory control, isolating the new 128K residual. Mean auxiliary gate mass on test was 0.13457; far evidence was available for 89.1% of sampled historical positions.

A smaller 256/128/128, 192-update confirmation also selected update 48: incremental dev gain +0.049748, test gain +0.207267, shuffled-history test advantage +0.251584, train-donor advantage +0.228034. Both results are *historical*, not independent new-book replications.

## Structural and safety checks

**12/12 passed:** correct remote value retrieval; 32K exclusion; 2K recent exclusion; target-change cannot affect retrieval features; no cross-band leakage; far context ignores near modifications; reproduce prior future-donor bug; shuffle preserves unigrams; target-free input shape; empty far memory exactly falls back to R7.16; zero eligible routes receive zero mass; and whole-book sampler obeys boundaries and equal-book coverage. Full parent and prerequisite adapter checksums were reverified after training. Explicitly no accepted checkpoint write, no optimizer reset and no FA2 changes.

## New-book promotion gate remains OPEN (not executed)

NaNROUTER's 3,615 English short-story books are published in GitHub Release archives, e.g. `corpora-short-stories-0001`, not directly in the repository source tree. The sandbox cannot fetch that binary asset for independent evaluation. The ZIP includes `fa1_r718_newbooks.py`, a read-only evaluator that checks whole-book hashes, known-token overlap screening against legacy corpus, document-boundary masking, equal-book bootstrap confidence intervals, and both independent counterfactual control types. It requires at least 12 age-eligible new books and positive lower confidence bounds for accepted-model gain, extra 128K gain, and both controls. Passing this gate would trigger **manual review**, never automatic promotion.

GitHub Actions is prepared for a sealed NaNROUTER evaluation but cannot execute until a separately hosted and checksum-verified **accepted R7.13 full archive** becomes available to its runner. The repo carries small source/adapter material, not the large accepted checkpoint.

## Next frontier-relevant engineering

1. **Real data first:** run the sealed book-disjoint evaluation and report word NLL and raw-byte BPB separately; do not claim meaningful new-book gains from old samples.
2. **Add an incremental sparse suffix index:** maintain 1–4-gram transitions and update age-band queues at O(1)–O(log N) amortized per token instead of rescanning 98K older tokens per sample; verify logit/p-target parity against the current exact rescan.
3. **Use semantic retrieval and revision tests:** paraphrased queries, temporal conflict, hard lexical distractors, 2-hop document QA. Exact matches alone are not semantic understanding.
4. **Integrate only after true book gates:** merge the successful adapter state and optimizer history atomically into the existing unified checkpoint and preserve exact fallback under gate-off.
5. **Scale model capacity** by identity-preserving 96→192→384 hidden migration and rank-64→128 mirror memory, evaluated on matched compute and held-out BPB. This is a proposed future stage, not yet implemented.

## Reproduce

Extract the verified full R7.13 archive into an FA1 working root, then layer this source/reports ZIP into that root. Install Python, NumPy, PyTorch (CPU), sentencepiece and zstandard for optional corpora import.

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$(pwd)/source"
python source/test_fa1_r718.py
python source/fa1_r718_far_residual.py --root . \
  --r716-overlay adaptation/fa1_r716_225.pt \
  --train 1024 --dev 512 --test 512 --steps 240 --eval-every 24 \
  --output reports/FA1_R7p18_1024_512_512.json \
  --overlay adaptation/fa1_r718_full_candidate.pt
```

Checksummed release includes source, all previous FA1 R7.14–R7.16 source/report history, R7.17 failed contrastive diagnostics, R7.18 selected small overlay, logs, metadata and one SHA manifest. No full parent checkpoint, nested release ZIP, corpus database, or FA2 artifacts.