# Z4 causal 84M — 269.19M tokens (18-hour cumulative)

Third 6-hour T4 block. This block switched to **fp16 AMP** and a corrected
**lr 1e-4 / decay-frac 0.5** WSD schedule.

| Metric | Value |
|---|---:|
| Total tokens | 269,193,856 |
| Tokens this block | 93,429,760 |
| Valid BPB | 1.2030 |
| Valid top-1 | 38.14% |
| Valid top-5 | 56.41% |
| **PTB BPB** | **1.6849** |
| PTB top-1 | 30.01% |
| PTB top-5 | 42.79% |
| Wall | 21,603.3 s |
| Throughput | 4,326 tok/s (fp16) |
| Non-finite skips | 5 / 22,810 steps |

Batch 8, ctx 512, T4. The fp16 path is 1.81× faster than bf16; the low-peak
schedule doubled the block's PTB gain relative to the 12h block.

**Caveat:** generation is locally fluent but globally incoherent; at
269M tokens / 84M parameters (3.2 tokens/param) the model is substantially
undertrained. PTB is the clean OOD metric.

**Kaggle mirror:** https://www.kaggle.com/datasets/shem42/z4-causal-gpu18h-weights

**Assets:** `z4-causal-84m-269m-final.pt`, `-best.pt`, `-config.json`,
`-result.json`, `-eval-ptb.json`, `-train.log`.

License: CC0-1.0.
