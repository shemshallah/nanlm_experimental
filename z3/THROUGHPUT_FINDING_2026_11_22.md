# Throughput breakdown: the sidecar is not the bottleneck (2026-11-22)

Measured with `shem42/z4-cpu-sidecar-bench` on a Kaggle CPU kernel, using the
same layered tokenizer and 512-token windows as training.

## Full batch (8 rows × 512 tokens, build + collate)

| Run | ms |
|---|---:|
| 1 | 210.0 |
| 2 | 244.0 |
| 3 | 205.9 |
| 4 | 203.3 |
| 5 | 203.6 |
| **Median** | **205.9** |

## Per-row components

| Component | ms/row | Share |
|---|---:|---:|
| `SpanSidecar.from_encoding` (tensor construction) | **21.68** | 86% |
| `word.encode_with_offsets` | 1.50 | 6% |
| `math.spans` | 0.20 | 1% |
| `_char_byte_offsets` | 0.21 | 1% |
| `subword.decode` | 0.05 | <1% |
| `sentence.split` | 0.03 | <1% |
| **Full `from_token_ids`** | **25.23** | 100% |

8 × 25.23 ms = 201.8 ms, plus ~4 ms for `collate`, matching the 205.9 ms
batch measurement.

## What it means

The T4 step is ~1.65–1.72 s for 4,096 tokens. The sidecar is ~0.206 s, or
**~12% of the step**. Even eliminating it entirely would cap the speedup at
~1.14×. The remaining ~1.45 s is GPU work and launcher overhead.

So:

- Do **not** spend effort on tokenizer/sidecar micro-optimization first.
- The main lever is the GPU path: fp16 on T4 (native tensor cores) versus
  the bf16 path both 6-hour runs forced.
- Secondary lever: occupancy. Batch 8 × 512 is small for a T4; batch 12–16
  would raise utilization if it fits. The training script already falls back
  to batch 4 on OOM, so memory headroom is unknown.
- If a numpy fast path for `from_encoding` is ever wanted, it is worth at
  most ~0.18 s/step (~10%); keep it as a later cleanup, not a priority.

The `z4-gpu-amp-probe` kernel tests the fp16 lever at identical batch/ctx.
