# FA1 R7.19 — Incremental causal 128K suffix-memory index

**Date:** 2026-11-12  
**Status:** CPU implementation and initial parity passed; **unpromoted experimental acceleration**, no new weights.  
**Accepted parent:** FA1 R7.13, step **14,416**, SHA-256 `54e835b00642013d4f1bf486a8150eb1ff06c547565771bd21933e143d050f2d`. Five inherited optimizer histories verified intact.  
**Fixed prerequisites:** frozen R7.16 candidate SHA-256 `436ff87fae69ef1f44b53e2ffde92313da408342f65a1897c6eaff1876f9c09c`, R7.18 far gate SHA-256 `f4d9175ff202d9ec1ff3a06dcddfd3d262cea25f8441fa5dab972a313309aa9f`.

## What changed

R7.18 computed 1–4-gram weighted n-gram statistics by rescanning all 98,304 far-band tokens on **each** prediction. R7.19 uses a causal streaming index with one indexed transition per order entering and leaving the 32K–128K band per token. It maintains per-key successor frequencies, exponentially age-decayed probability mass, successor entropy, latest-match recency and a lazily compacted heap for the modal successor. The feature semantics, gate weights, target label isolation, disjoint 32K near band and 2K recent band are preserved. The index never reads the target or future positions. There is no new semantic memory or model capacity.

- Initial construction: `O(4 * 98,304)` transition updates; measured 1.18–2.50 seconds on synthetic 128K sequences.
- Streaming step + four-path lookup: amortized `O(4 log U)` heap operations and fixed per-order insertions/expirations, where U is the number of unique successor values for affected keys. Rare heap rebuilds keep stale entries bounded.
- Auxiliary memory: `O(4 * 98,304)` indexed transitions. Python dictionaries/heaps may consume substantial RAM; a packed C++ implementation is a separate future optimization.
- Arbitrary random access: not accelerated by this implementation; a distant seek must rebuild from history. Only sequential/near-sequential inference amortizes the build cost.
- Reference uses float32 exponential age weights; online index uses higher-precision lazy updates. Record values agree to float tolerance rather than bit-for-bit. The final gate/NLL test showed exact float32 outputs on its sampled examples.

## Local CPU measurement (synthetic, NOT whole-model throughput)

Original R7.18 reference rescan averaged **2.6075 ms/query**, while the new maintained index averaged **0.0578 ms per advance+query**, giving **45.14x speedup excluding initial construction** on a Zipf-distributed test stream. This benchmark used 16 independent reference queries and 100 maintained-index queries. It is *not* evidence of a 45x language-model or end-to-end decoding speedup. At this build cost, roughly 1,000 sequential queries are required before the accelerator wins on total time in the tested distribution. Additional RAM-vs-latency testing is required for Android.

## Validation

- **11/11** new R7.19 structure, causality, boundary, mode/entropy, target independence and synthetic NLL/gate parity tests passed.
- **12/12** inherited R7.18 regression tests passed.
- Full R7.13 archive (214.5 MB ZIP, containing the 94.65 MB checkpoint) SHA-256 was independently verified: `99f25e5ff604492c6c043335eddbf638415a423f43d94f5c9a426f5a283a2728`. Restored checkpoint SHA-256 `54e835b00642013d4f1bf486a8150eb1ff06c547565771bd21933e143d050f2d`. R7.16/R7.18 adapter hashes verified.
- **4 actual historical test positions**: reference R7.18 and R7.19 indexed predictor produced the same float32 gate outputs and cross-entropy. The maximum absolute input-feature discrepancy was `7.45e-9`; gate and NLL discrepancy were both `0.0` on this sample. This is a small parity smoke test, not a broad generalization test. A larger 16-position test exceeded the local CPU execution budget; it is not a successful run.
- Historical test loss from the first four positions is *not* a new language-model gain; this release solely targets computation speed while preserving R7.18 behavior.

## Independent NaNROUTER books: still blocked

No independent new-book evaluation was performed and **no checkpoint is promoted**. The existing `fa1-r718-newbooks.yml` can verify new books once the full R7.13 archive is available as the expected GitHub Release asset. The GitHub tag `fa1-accepted-r7.13` is absent at this writing. GitHub currently contains source and small adapter files, not the full accepted checkpoint. The R7.18 book-level bootstrap criteria and the two corrected history controls remain required before any promotion.

## How to reproduce

Install Python 3.11+, NumPy, CPU PyTorch, and extract the **R7.13 full archive** to `ROOT` without changing its content. Put the R7.18 small release's `source/` on `PYTHONPATH`, followed by `ROOT/source`:

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$(pwd)/source:$ROOT/source"
python source/test_fa1_r718.py
python source/test_fa1_r719.py | tee reports/FA1_R7p19_PARITY_BENCHMARK.json
python source/test_fa1_r719_real_checkpoint.py \
  --root "$ROOT" --near adaptation/fa1_r716_225.pt \
  --far adaptation/fa1_r718_full_candidate.pt --examples 4 \
  | tee reports/FA1_R7p19_REAL_CHECKPOINT_PARITY.json
```

**Known limitation:** while the read-only accelerator is implemented and tested, the R7.18 `gather` training/evaluation function still uses its original random-position rescan. This release does not claim that all FA1 processing is indexed. Incremental indexing is appropriate for online contiguous prediction or a new streaming evaluator with explicit document-state resets.

**Next milestones:** incorporate the indexed reader into a document-streaming evaluator (not random historical sampler), measure memory peaks and full-token inference speed, run at least 12 independently verified long NaNROUTER books with corrected chronological and train-donor controls, and only then assess promotion of the frozen adapter. Test any future semantic/revision/multi-hop changes as distinct candidate architectures rather than attributing those capabilities to this exact n-gram retriever.