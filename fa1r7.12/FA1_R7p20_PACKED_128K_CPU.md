# FA1 R7.20 — packed causal 128K CPU memory

**Date:** 2026-11-12  
**Lineage:** FA1 only. FA2/NaNLMind/Poincaré Mirror LM are separate.  
**Status:** Verified **read-only acceleration**, **not promoted**.  
**Accepted FA1:** step 14,416, checkpoint SHA-256 `54e835b00642013d4f1bf486a8150eb1ff06c547565771bd21933e143d050f2d`; five original optimizer histories retained.  
**R7.16 near overlay SHA-256:** `436ff87fae69ef1f44b53e2ffde92313da408342f65a1897c6eaff1876f9c09c`  
**R7.18 far overlay SHA-256:** `f4d9175ff202d9ec1ff3a06dcddfd3d262cea25f8441fa5dab972a313309aa9f`

## Finding: faster only when the full workload justifies the index

R7.19 replaced full historical scans with an incrementally maintained Python dictionary of decaying n-gram statistics. Its synthetic post-construction microbenchmark was fast, but constructing ~393K per-event Python objects took ~3 seconds and required substantial process memory. On **four** real-checkpoint positions, R7.19 was **slower**, despite exact NLL parity. On **256** positions, the R7.19 history phase was ~3.18 seconds vs ~0.28 seconds for the original R7.18 rescanning algorithm: an important adverse finding that was not visible in the first microbenchmark.

R7.20 replaces that implementation with a **collision-free packed** `uint64` n-gram key (up to 4 x 15 bits for vocab IDs 0–32,767), together with sorted NumPy key and occurrence arrays. Its stored base arrays occupy ~4,718,472 bytes for a full 128K history window. The NumPy implementation checks token range and fails rather than silently aliasing out-of-range IDs. An ordered, bounded pending-key dictionary stores freshly eligible causal transitions; periodic rebuilds discard expired positions and compact the arrays. Queries explicitly filter out every position older than 131,072 tokens, younger than 32,769 tokens, or before the active book boundary. Query features, modal selections and prior odds do not depend on the true target. The true target is used **only** for scoring the held-out next-token probability.

`fa1_r720_runtime.py` integrates this packed index into **actual end-to-end** frozen FA1 R7.13 + R7.16 + R7.18 scoring. `exact_reference` rescans R7.18 far history; `packed_index` uses the packed state; `auto` uses exact rescans on short/sparse requests and packed state on long consecutive same-document streams. Crucially, R7.12's inherited 6,144-token mirror history is still block-initialized per batch, matching the historical R7.18 evaluator. R7.12's *different* per-token mirror recurrence is **not** silently substituted for that block initializer. The 128K module remains sparse exact 1–4-gram retrieval, **not** learned semantic attention across 128K tokens.

## Measured real-model CPU parity (same historical test sample)

Two separate runs each compared **1,024 consecutive positions**, starting at historical test position 134,000, batches of 16, on the verified accepted FA1 checkpoint with both frozen candidate overlays. Model scoring and index construction are included. The two runs reverse the order of the reference and packed modes to reduce simple warm-cache order confounding.

| Measured quantity | Reference-first run | Packed-first run |
|---|---:|---:|
| Original R7.18 total runtime | 10.1326 s | 10.9677 s |
| Packed R7.20 total runtime | 8.6670 s | 9.4233 s |
| Total speedup | **1.1691x** | **1.1639x** |
| R7.18 far-memory stage | 1.07595 s | 1.11655 s |
| R7.20 packed far-memory stage (build included) | 0.26984 s | 0.28380 s |
| Maximum final NLL absolute difference | 9.54e-7 | 9.54e-7 |

The first run averaged 5.435134 nats/word for the inherited baseline, 5.235852 for frozen R7.16, and 5.145802 for both R7.18 and R7.20. These are **nats per word on a reused historical split**, not bits per byte or a generalization benchmark. A separate four-position cross-check against the original R7.18 `gather` implementation found *exact* baseline, near and final NLL agreement. The two execution orders have ordinary run-to-run CPU variance and should not be mistaken for a matched-sampling statistical confidence interval.

## Structural and benchmark tests

* Packed index: **10/10** tests on constant, periodic and Zipf-token streams, expiration/rebuild, 32K/128K age bounds, book-start boundaries, target-feature isolation and 15-bit capacity fail-closed.
* Inherited R7.19: **11/11** tests passed in the previous round.
* Inherited R7.18: **12/12** structural tests passed in the previous round.
* Packed index initial arrays: **4.7 MB**; process memory additionally includes the PyTorch model, corpus and working buffers.
* Synthetic 700-query workload: **1.20x** faster including index construction and **1.21x** excluding construction, for that particular Zipf distribution. Do **not** extrapolate this directly to end-to-end language-model throughput.

## Still blocking model promotion

No independent whole-book NaNROUTER R7.18/R7.20 gate has run in this environment. The NaNROUTER batch `corpora-short-stories-0001` exists, but its ~110 MB release asset cannot be fetched into the current local container; GitHub Actions still lacks the **accepted R7.13 full checkpoint release** at `fa1-accepted-r7.13`. The restored full R7.13 archive remains separately verified in the ChatGPT Library, not GitHub Releases. R7.20 does **not** change the 128K reader's predictive ability; it only accelerates the same reader.

The companion manual workflow `fa1-r720-newbooks.yml` is staged to download SHA-pinned sources, verify original checkpoint and overlays, import independent books, perform the existing R7.18 book-cluster/negative-control gating, and additionally prove R7.20-vs-R7.18 parity within 12 whole books. It must **fail closed** while the checkpoint release is absent; no automatic promotion is performed.

## CPU reproducibility

```bash
export PYTHONPATH="$PWD/source:$FA1_FULL_ROOT/source"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
python source/test_fa1_r718.py
python source/test_fa1_r719.py
python source/test_fa1_r720.py
python source/test_fa1_r720_real_checkpoint.py --root "$FA1_FULL_ROOT" \
  --near adaptation/fa1_r716_225.pt --far adaptation/fa1_r718_full_candidate.pt \
  --positions 1024 --batch 16 --output reports/R720_REAL_CHECKPOINT_1024.json
python source/test_fa1_r720_real_checkpoint.py --root "$FA1_FULL_ROOT" \
  --near adaptation/fa1_r716_225.pt --far adaptation/fa1_r718_full_candidate.pt \
  --positions 1024 --batch 16 --reverse --output reports/R720_REAL_CHECKPOINT_1024_REVERSE.json
```

The archive excludes the full parent checkpoint, corpus DB and binaries. All inherited optimizer histories and model tensors remain untouched. Continue FA1 R7.21 by verifying split-disjoint whole books, then explore semantic keys, adaptive memory routing and full-model CPU throughput under nonleaking controls. Do not claim frontier-level capability without those evaluations.