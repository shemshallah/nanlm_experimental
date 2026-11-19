# Broad non-WikiText 1-hour run (2026-11-19)

Fresh 115,238,237-parameter model, trained on a broad corpus with no
WikiText or Wikipedia collection.  Batch 64, 68,812,800 tokens, 2,100
steps, 3,265.4 s (54m 25s), 0 non-finite skips.

## Corpus

- 15 NaNROUTER domain releases: arXiv, PubMed, law, patents, code, math,
  news, Q&A, recipes, security, standards, humanities, chemistry, finance
- FineWeb sample (70 MB), TinyStories (40 MB), AG News, word/spelling lists
- Documents chunked to ~1 KB to avoid the tokenizer's superlinear
  long-document cost

| Split | Tokens | Documents |
|---|---:|---:|
| train | 125,489,451 | 620,376 |
| valid | 1,273,703 | 6,280 |
| test | 1,301,188 | 6,421 |

## Final evaluations

| Eval | Tokens | BPB | Top-1 |
|---|---:|---:|---:|
| In-domain valid (chunk-split) | 1,273,703 | 0.2171 | 88.45% |
| Official WikiText-2 test | 293,652 | 0.3684 | 79.32% |
| Official WikiText-103 test | 282,936 | 0.3265 | 80.13% |
| Tiny Shakespeare | 337,118 | 0.5188 | 73.30% |
| **PTB test (clean OOD)** | **105,171** | **0.8679** | **50.84%** |
| Random-word sanity check | 383,270 | 1.1727 | 27.76% |

## The honest interpretation

- **PTB is the clean under-1.0 result.** PTB is 1980s Wall Street Journal,
  pre-web, not in FineWeb/Gutenberg/AG News. 32-token overlap against a
  5M-token train sample is **0/105,141**. This meets the “different domain,
  under 1.0” bar.
- **WikiText/Shakespeare are not clean zero-shot.** The broad corpus
  contains web/Gutenberg material: the source slices contain strings
  “Romeo”, “Hamlet”, and “Macbeth”, and FineWeb can contain Wikipedia-like
  text. The 0.37/0.33/0.52 numbers are therefore domain-overlap results,
  not clean zero-shot claims.
- The full 32-token shingle check against the **full 125M train stream**
  was 0 for WikiText-2, WikiText-103, and Shakespeare, but that only rules
  out exact token sequences, not same-work/different-edition overlap.
- The in-domain 0.2171 is optimistic: 1 KB chunking can split adjacent
  chunks of the same source document across train/valid.
- No-sidecar evaluation is identical, so the low BPB is not sidecar
  leakage.
- The subword tokenizer is WikiText-2-derived.  This is model-weight
  zero-shot, not strict tokenizer-level zero-shot.
- No leaderboard claim: the standard benchmarks train on their own splits.

## Blackwell optimizations

- Batch 64 is the safe maximum: ~19.9k tokens/s. Batch 96 OOMs at ~85 GB.
- Fused AdamW enabled (with fallback).
- Dense MoE dispatch, bf16, no gradient checkpointing, 8 threads.
- Optional `torch.compile` implemented but not used.
- 1 KB document chunking took tokenizer throughput from ~0.02–0.06 MB/s
  on long finance/FineWeb documents to ~0.48–0.54 MB/s.

## Reproduction

```sh
python z3/train_v3.py \
  --device cuda --amp --amp-dtype auto \
  --preset kaggle10m_nomtp \
  --corpus data/broad_bins/train.bin \
  --val-corpus data/broad_bins/valid.bin \
  --data-manifest data/broad_bins/manifest.json \
  --batch-size 64 --ctx 512 --lr 4e-4 \
  --steps 2100 --lr-total-steps 2100 \
  --eval-every 500 --eval-batches 2 \
  --max-seconds 3600 --no-grad-checkpoint --threads 8 \
  --moe-dense-dispatch \
  --out runs/broad_hour
```

PTB clean OOD eval:

```sh
python z3/eval_domains.py \
  --checkpoint runs/broad_hour/best.pt \
  --test-dir data/ood_ptb_chunked \
  --tokenizer layered --document-framing --sidecar --require-sidecar \
  --ctx 512 --batch-size 16 --device cuda \
  --out data/broad_hour_ptb.json
```

The next clean WikiText claim needs a training corpus without FineWeb,
Gutenberg/humanities, Wikipedia, or Shakespeare.  The domain mix
(code/math/news/arXiv/PubMed/law/patents/Q&A/recipes/security/standards/
chemistry/finance + TinyStories/AG News/word lists) is the basis for that
run.
