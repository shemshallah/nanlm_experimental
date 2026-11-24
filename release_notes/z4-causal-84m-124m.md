# Z4 causal 84M — 124.18M tokens (6-hour block)

First long GPU continuation of the causal 84,455,112-parameter Z4, resumed
from the 70.5M-token pilot checkpoint.

| Metric | Value |
|---|---:|
| Total tokens | 124,179,072 |
| Tokens this block | 53,678,080 |
| Valid BPB | 1.2839 |
| Valid top-1 | 35.81% |
| **PTB BPB** | **1.7624** |
| PTB top-1 | 28.48% |
| PTB top-5 | 40.34% |
| Wall | 21,604.9 s |
| Non-finite skips | 0 |

Batch 8, ctx 512, bf16 AMP, T4. PTB is the clean OOD benchmark; per-domain
numbers are in `z3/CAUSAL_GPU6H_2026_11_21.md`.

**Kaggle mirror:** https://www.kaggle.com/datasets/shem42/z4-causal-gpu6h-weights

**Assets:** `z4-causal-84m-124m-final.pt`, `-best.pt`, `-config.json`,
`-result.json`, `-eval-ptb.json`, `-train.log`.

License: CC0-1.0.
