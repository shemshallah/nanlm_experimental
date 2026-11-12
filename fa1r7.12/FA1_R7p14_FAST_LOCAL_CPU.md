# FA1 R7.14 — Fast local CPU test (no promotion)

**Lineage:** FA1 only; FA2 untouched. **Accepted model:** 14,416 joint updates, five optimizer histories. **Checkpoint SHA-256:** `54e835b00642013d4f1bf486a8150eb1ff06c547565771bd21933e143d050f2d`.

The original R7.13 full recovery archive was materialized locally and verified: `FA1_R7p13_HIERARCHICAL_SINGLE_CHECKPOINT_FULL.zip` (214,538,430 bytes; SHA-256 `99f25e5ff604492c6c043335eddbf638415a423f43d94f5c9a426f5a283a2728`). Its accepted full checkpoint SHA matched the R7.13 manifest before and after all runs. Original trained LM parameters and all five optimizer histories remained unchanged.

## Fast test outcomes

| CPU experiment | Steps | Train/dev/test | Unmodified LM dev NLL ↓ | Candidate dev NLL ↓ | Correct vs wrong-memory advantage | Outcome |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| A. R7.13 hierarchical readout + residual trust gate | 128 attempted; best adapter step 16 | 128 / 64 / 64 | 4.703228 | 4.704876 | -0.004810 | **Rejected** |
| B. Causal 1/2/3-gram historical-cache comparator | 160 attempted; no accepted step | 256 / 128 / 128 | 5.296993 | 5.296993 (cache disabled) | 0.000000 after fallback | **Rejected** |

Both experiments used historical, previously consulted train/dev/test corpora, not sealed author/book-disjoint evaluations; these numbers are **nats per target word**, not raw bits/byte. Different sampled dev positions mean A and B NLLs are not directly comparable.

A reused the existing R7.12 calibration gate and R7.13 8×768 hierarchical directed-memory reader. The new adapter initialized to identical predictions; a new optimizer parameter group was appended to the **existing fifth optimizer**, with no modifications to inherited slots. A improved vs the inherited hierarchy's 4.705755 dev NLL, but not vs the full LM, and *unrelated* history still predicted better than true history. On historical test: base LM 5.746750 NLL; A candidate 5.745140, but wrong-memory 5.704425. **Do not promote.**

B independently tested a strict causal 3→2→1-token exact-cache backoff with recency weighting. It is a **nonparametric control**, not a replacement for FA1's learned rank-64 mirror memory. Initial cache dev NLL was 5.297455 vs base 5.296993, and wrong-memory NLL 5.265017. The intermediate dev NLL of 5.296909 at step 16 did not pass the wrong-memory gate. Its selected fallback was no cache; historical test NLL equals the unchanged LM at 5.973211.

The inherited R7.13 32-query synthetic regression also passed causal-routing, temporal-conflict and embedding-neighbor probes. No new full-model weight checkpoint was promoted.

## Reproducibility

A minimal 15,557-byte source/reports ZIP was produced in the research conversation:
`FA1_R7p14_FAST_LOCAL_CPU_SOURCE_REPORTS.zip`; SHA-256 `99c98c58596391c6364fbb0cc7212d91d2438a0ae1b2cc80eae6ed71ae12ceab`.

It contains exact scripts `source/fa1_r714_gate.py`, `source/fa1_r714_cache_probe.py`, 3 JSON reports, a method/limitations README, manifest and SHA256SUMS. It deliberately excludes checkpoints, candidate weights, databases, and corpora; it requires the **actual** R7.13 full release for reruns. The two rejected experimental adapter payloads are deliberately not included.

CPU commands after overlaying scripts on the verified full R7.13 release:

```bash
PYTHONPATH=source OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python source/fa1_r714_gate.py --root . --train 128 --dev 64 --test 64 --updates 128 --threads 2
PYTHONPATH=source OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python source/fa1_r714_cache_probe.py --root . --train 256 --dev 128 --test 128 --updates 160 --threads 2
```

**Next gate:** train a joint query-conditioned memory-to-logits bridge, evaluate genuine correct-vs-unrelated history gain and raw text BPB on independent NaNROUTER books, preserve the unique checkpoint and all optimizer histories until an actual promotion gate passes.

**Publication limitation:** This GitHub text note contains experiment metrics and provenance. The runnable new scripts and JSON outputs are in the small local ZIP above, **not** in this GitHub subtree yet. It is not an automatically synced release.
