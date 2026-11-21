# Causal GPU 6h: PTB and per-domain results (2026-11-21)

6-hour Kaggle T4 continuation of the causal 84M model. Resumed the
70.5M-token checkpoint and trained 53,678,080 more tokens (124.18M total).

## Run

| Field | Value |
|---|---|
| Parameters | 84,455,112 |
| Steps this run | 13,105 |
| Tokens this run | 53,678,080 |
| Total tokens | 124.18M |
| Wall | 21,604.9 s (6h 00m) |
| Throughput | ~2.5k tokens/s |
| Batch / ctx | 8 / 512 |
| Non-finite skips | 0 |
| Status | completed (`max_seconds`) |

## Clean OOD and validation

| Metric | 70.5M | **124.18M** |
|---|---:|---:|
| In-domain valid BPB | 1.3165 | **1.2839** |
| Valid top-1 | 32.35% | **35.81%** |
| **PTB BPB** | 1.8928 | **1.7624** |
| **PTB top-1** | 25.63% | **28.48%** |
| PTB top-5 | 36.99% | **40.34%** |

## Per-domain held-out BPB

`test.bin` slices, up to 1M tokens per group. Chunk-level held-out, same
source; optimistic relative to external tests.

| Domain | BPB | Top-1 | Top-5 | 70.5M BPB |
|---|---:|---:|---:|---:|
| FineWeb-Edu | 1.3382 | 28.80% | 48.37% | 1.4268 |
| Code | 1.0865 | 48.70% | 68.98% | 1.1980 |
| Math | 1.0842 | 48.92% | 70.15% | 1.1943 |
| NaNROUTER news | 1.5261 | 24.29% | 43.11% | 1.6291 |
| AG News | 1.5026 | 28.48% | 44.67% | 1.6353 |
| Cosmopedia | 0.9998 | 33.82% | 52.62% | 1.0866 |
| arXiv | 1.2572 | 32.89% | 51.91% | 1.3504 |
| PubMed | 1.3990 | 31.17% | 48.02% | 1.5172 |
| TinyStories | 0.9880 | 43.44% | 68.78% | 1.0945 |
| FineWeb-70M | 1.5220 | 25.99% | 44.82% | 1.6118 |
| Law | 1.4393 | 28.85% | 47.45% | 1.5283 |
| Patents | 1.1491 | 33.96% | 52.30% | 1.2640 |
| Finance | 1.1580 | 31.22% | 50.75% | 1.2611 |
| Security | 1.3789 | 34.46% | 52.51% | 1.4678 |
| Standards | 1.5360 | 27.95% | 45.48% | 1.6213 |
| Chemistry | 0.9871 | 39.45% | 59.89% | 1.0965 |
| Recipes | 1.3480 | 33.40% | 53.90% | 1.4773 |
| QA | 1.3591 | 27.76% | 46.95% | 1.4388 |

Every domain improved by 0.08–0.13 BPB. FineWeb-Edu, Code, Math, and
NaNROUTER news are 1.34, 1.09, 1.08, and 1.53 respectively. The table is
tight enough for a same-corpus universality check; PTB remains the only
clean OOD number.

## Persistence

- Weights: `shem42/z4-causal-gpu6h-weights`
  (`final.pt`, `best.pt`, `config.json`, `result.json`, `eval_ptb.json`,
  `train.log`)
- Per-domain JSON: `per_domain_gpu6h.json` in the Kaggle kernel output
  `shem42/z4-gpu6h-domain-eval`

## Note on throughput

The T4 path only processed ~2.5k tokens/s, so 6 hours covered 5.6% of the
955M corpus. The corpus was not the limiter; the single-T4 implementation
was. Blackwell or TPU would be needed for a materially larger token budget.
