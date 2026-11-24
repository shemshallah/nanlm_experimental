# Z4 causal 84M — 175.76M tokens (12-hour cumulative)

Second 6-hour T4 block, resumed from the 124.18M-token checkpoint.

| Metric | Value |
|---|---:|
| Total tokens | 175,764,096 |
| Tokens this block | 51,585,024 |
| Valid BPB | 1.2488 |
| Valid top-1 | 36.86% |
| Valid top-5 | 54.75% |
| **PTB BPB** | **1.7349** |
| PTB top-1 | 29.04% |
| PTB top-5 | 41.01% |
| Wall | 21,604.0 s |
| Non-finite skips | 0 |

Batch 8, ctx 512, bf16 AMP, T4. Full 1M-token per-domain table:
`z3/CAUSAL_GPU12H_DOMAINS_2026_11_22.md`.

**Kaggle mirror:** https://www.kaggle.com/datasets/shem42/z4-causal-gpu12h-weights

**Assets:** `z4-causal-84m-176m-final.pt`, `-best.pt`, `-config.json`,
`-result.json`, `-eval-ptb.json`, `-train.log`.

License: CC0-1.0.
