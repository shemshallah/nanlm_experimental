# 12h checkpoint: interim per-domain table on CPU (2026-11-22)

The GPU per-domain eval was blocked by the large-input mount incident, so the
same 12h checkpoint (`step 30002`, 84,455,112 params, causal) was evaluated on
a Kaggle CPU kernel.

## Protocol

- Device: CPU, fp32 (no AMP)
- Token cap: **200,000 per domain group** (GPU table used up to 1,000,000)
- Same `test.bin` slices, causal sidecar, chunk-level held-out
- Actual token counts vary where a group has less test data (e.g. NaNROUTER
  news 40,880; PubMed 29,127; Standards 9,709)
- Wall: 5,060 s for all 18 groups

Because the cap and sample are smaller than the 6h GPU run, the absolute
values are **not directly comparable** to the 1M-token table. The direction
and ranking are still useful.

## Results

| Domain | BPB | Top-1 | Top-5 | 6h GPU BPB (1M cap) |
|---|---:|---:|---:|---:|
| FineWeb-Edu | 1.3053 | 29.56% | 49.42% | 1.3382 |
| Code | 0.9102 | 55.00% | 75.58% | 1.0865 |
| Math | 1.0489 | 50.12% | 71.32% | 1.0842 |
| NaNROUTER news | 1.4877 | 25.21% | 44.42% | 1.5261 |
| Cosmopedia | 0.9669 | 34.94% | 54.13% | 0.9998 |
| arXiv | 1.2209 | 34.02% | 53.42% | 1.2572 |
| PubMed | 1.3635 | 32.30% | 49.49% | 1.3990 |
| AG News | 1.4528 | 29.99% | 46.64% | 1.5026 |
| TinyStories | 0.9397 | 45.21% | 70.82% | 0.9880 |
| FineWeb-70M | 1.4907 | 26.77% | 46.01% | 1.5220 |
| Law | 1.4070 | 29.95% | 48.77% | 1.4393 |
| Patents | 1.1156 | 34.81% | 53.54% | 1.1491 |
| Finance | 1.1257 | 31.86% | 52.10% | 1.1580 |
| Security | 1.3282 | 35.78% | 54.38% | 1.3789 |
| Standards | 1.5058 | 28.65% | 46.64% | 1.5360 |
| Chemistry | 0.9384 | 41.03% | 62.16% | 0.9871 |
| Recipes | 1.2924 | 35.06% | 56.19% | 1.3480 |
| QA | 1.3311 | 28.73% | 48.27% | 1.3591 |

Every group improved versus the 6h checkpoint. The largest apparent drop is
Code (1.0865 → 0.9102), but that group is a concatenation of several code
sources and the 200k cap samples only the front of the group, so the
composition differs from the 1M-token GPU pass. Treat the Code delta as
provisional.

PTB remains the clean OOD number: **1.7349 BPB / 29.04% top-1 / 41.01% top-5**
at 175.76M tokens.

## Persistence

- Kernel: `shem42/z4-cpu-domain-eval-12h`
- JSON: `per_domain_cpu12h.json` in kernel output (download was rate-limited
  at the time of writing; values above are from the kernel log)
- Full protocol will be repeated on GPU when large mounts recover
