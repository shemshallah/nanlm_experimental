# 84M 30-minute checkpoint and per-domain held-out BPB (2026-11-19)

20-minute continuation from the 81.1M-token 84M checkpoint, batch 64,
ctx 512, time-based WSD, 1,200 s. Total across runs: **105.6M tokens**.

## Run

| Field | Value |
|---|---|
| Parameters | 84,455,112 |
| Tokens this run | 24,510,464 |
| Total tokens | 105.6M |
| Wall time | 1,206.9 s |
| Throughput | ~20.3k tokens/s |
| In-domain valid | **0.1596 BPB / 89.38% top-1** |
| **PTB test (clean OOD)** | **0.7850 BPB / 54.62% top-1 / 72.40% top-5** |

PTB improved from 0.9616 at 81.1M tokens to **0.7850** at 105.6M. The
in-domain valid number is optimistic.

## Per-domain held-out BPB

Source: document-framed `test.bin` slices, up to 1M tokens per group.
These are **chunk-level held-out**, not article-level held-out.

| Domain | BPB | Top-1 | Top-5 | Tokens |
|---|---:|---:|---:|---:|
| **FineWeb-Edu** | **0.1209** | 92.00% | 96.18% | 999,936 |
| **Code** (Glaive + SO + NaNROUTER) | **0.2157** | 87.19% | 94.86% | 707,584 |
| **Math** (FineMath + NaNROUTER) | **0.2191** | 87.32% | 95.16% | 999,936 |
| **NaNROUTER news** | **0.1086** | 94.48% | 96.46% | 40,960 |
| AG News | 0.4001 | 71.84% | 87.05% | 52,224 |
| Cosmopedia | 0.0704 | 94.20% | 97.16% | 999,936 |
| arXiv | 0.1098 | 91.61% | 96.91% | 138,240 |
| PubMed | 0.1721 | 87.86% | 95.24% | 29,184 |
| TinyStories | 0.0977 | 93.82% | 96.92% | 52,224 |
| FineWeb-70M | 0.1385 | 91.48% | 95.78% | 147,456 |
| Law | 0.1798 | 88.87% | 94.52% | 23,040 |
| Patents | 0.1581 | 86.33% | 95.01% | 18,944 |
| Finance | 0.1554 | 87.63% | 94.75% | 22,016 |
| Security | 0.1982 | 86.95% | 94.27% | 15,360 |
| Standards | 0.2286 | 84.72% | 93.35% | 9,728 |
| Chemistry | 0.0804 | 94.37% | 97.37% | 15,872 |
| Recipes | 0.0357 | 98.61% | 99.11% | 21,504 |
| QA | 0.1622 | 88.19% | 94.82% | 21,504 |

## What this does and does not prove

The table is **tight**: excluding AG News, every domain is 0.036–0.229 BPB,
and the four requested domains are 0.109–0.219. The model is not
specializing into one domain at the expense of the others.

But this is **not clean zero-shot universality**. The corpus was split by
**chunk hash**, so neighbouring ~1.2 KB chunks from the same original
article can land in both train and test. Recipes at 0.036 and Cosmopedia
at 0.070 are red flags for near-duplicate chunks. The table measures
within-corpus fit across domains; PTB (0.785) is the only clean OOD
number.

For a true universality table, the next step is an **article-level split**
(or external per-domain test sets) so no original document contributes
chunks to both sides. That is a separate corpus build; the current table
should be labeled same-source/chunk-held-out, not zero-shot.

## Persistence

- Kaggle: `shem42/z4-broad30m-84m`
  - `final.pt`, `best.pt`, `config.json`, `result.json`, `train.log`,
    `eval_ptb.json`, `per_domain_test.json`
- Corpus: `shem42/z4-broad7h-bins`
- Earlier checkpoints: `z4-broad20m-balanced`, `z4-broad10m-84m`
