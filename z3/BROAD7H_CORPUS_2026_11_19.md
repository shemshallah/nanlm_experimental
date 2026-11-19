# Broad 7-hour corpus and Blackwell run (2026-11-19)

The 7-hour run uses a **955M-token** curated corpus, tokenized with the
`layered_v1` artifact and packaged on Kaggle. It resumes the 40-minute
checkpoint and trains for 7 wall-clock hours.

## Why this mixture

The first 1.19B-token build came out **22.5% FineMath / 13.9% Cosmopedia**,
which is too math-heavy for a general model. The final corpus is
rebalanced by truncating those sources before merging:

| Source | Tokens | Share | Role |
|---|---:|---:|---|
| FineWeb-Edu (6 × 505 MB) | 622M | 65.1% | general educational web text |
| FineMath (`math_3B` part 1) | 100M | 10.5% | math reasoning |
| Cosmopedia v2 part 1 | 100M | 10.5% | synthetic textbook/explanatory prose |
| Glaive Python + StackOverflow 2020 | 67M | 7.0% | code and technical Q&A |
| FineWeb 70 MB + arXiv abstracts | 24M | 2.5% | diverse web and science |
| TinyStories + AG News | 10M | 1.0% | simple narrative and news |
| 13 NaNROUTER domains | 30M | 3.1% | law, patents, finance, security, standards, chemistry, recipes, Q&A, news, PubMed, arXiv, code, math |

Total: **954,899,229 train tokens**, 12,154,535 valid, 12,145,279 test.
Documents are chunked to ~1.2 KB and document-framed; duplicates are
removed per source; PTB/WikiText are not included.

The 7-hour run samples random windows, so it will see roughly **530M
unique tokens** at the measured ~21k tok/s — about 55% of the corpus,
with no epoch repetition. The remaining tokens are spare capacity for a
later run.

## Kaggle datasets

- `shem42/z4-broad7h-bins` — `train.bin`, `valid.bin`, `test.bin`,
  `manifest.json`, and `tokenizer/layered_tokenizer.json` (v2 is the
  balanced 955M corpus).
- `shem42/z4-broad7h-checkpoints` — hourly rolling `last.pt` plus
  `config.json`, `result.json`, `train.log`.

The tokenizer is inside the corpus dataset, so a fresh machine can run:

```sh
kaggle datasets download -d shem42/z4-broad7h-bins --unzip -p data/broad7h
python z3/train_v3.py \
  --preset kaggle10m_nomtp \
  --layered-model data/broad7h/tokenizer/layered_tokenizer.json \
  --corpus data/broad7h/train.bin \
  --val-corpus data/broad7h/valid.bin \
  --data-manifest data/broad7h/manifest.json \
  --batch-size 64 --ctx 512 --lr 4e-4 \
  --steps 30000 --lr-total-steps 30000 --lr-horizon time \
  --max-seconds 25200 --warmup-steps 30 --decay-frac 0.1 \
  --chunked-ce --ce-chunk-tokens 2048 \
  --moe-dense-dispatch --no-grad-checkpoint --no-ema \
  --eval-every 1000 --save-every 1000 --best-every 1000 \
  --out runs/broad7h
```

## Blackwell measurements

| Config | Tokens/s | Peak VRAM | Notes |
|---|---:|---:|---|
| dense MoE, full logits, batch 64 | 19,304 | 85.6 GB | original |
| dense MoE, chunked CE, batch 64 | 19,332 | 62.5 GB | adopted |
| sparse MoE, full logits, batch 64 | 20,416 | 79.1 GB | +5%, not worth the routing change |
| dense, chunked CE, batch 96 | 19,200 | fits | no throughput gain |
| dense, chunked CE, batch 128 | OOM | >95 GB | trunk activations still too large |
| `torch.compile` default | ~48 s/step | — | recompiles on dynamic MoE; unusable |

Two safe code changes are now in `z3/v3_model.py`:

- when `ctx <= window`, local layers use the causal SDPA fast path instead
  of a 512×512 bool mask;
- GQA uses `enable_gqa=True` instead of materializing 5× repeated KV.

The real memory fix is `z3/train_v3.py --chunked-ce`: the 64k head is
applied to 2,048-token chunks inside activation checkpointing, so the
2.1B-element logits tensor is never materialized. Loss and gradients match
the full path to ~1e-7.

Throughput is still ~20k tok/s, so 7 hours is ~530M tokens. The corpus is
larger than that on purpose; it avoids repeating documents within the run.

## Run status

Started 2026-11-19 04:58 UTC, ETA 11:58 UTC. Batch 64, ctx 512, time-based
WSD (stable to 90%, decay over the last 42 minutes), chunked CE, dense
MoE, sidecar on. Training loss fell from 1.94 at step 1410 to 1.46 at
step 1550.

Post-run: PTB test BPB/top-1, per-domain held-out BPB, and generation with
top-k 50 / top-p 0.9 / repetition penalty 1.1.
