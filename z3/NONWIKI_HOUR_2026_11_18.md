# Non-WikiText 1-hour run — zero-shot WikiText (2026-11-18)

Fresh 115,238,237-parameter model, trained **without any WikiText text**.
The only WikiText exposure is the fixed subword tokenizer vocabulary, which
was originally learned from WikiText-2; the model weights are zero-shot.

## Corpus (no WikiText)

- **Word lists:** `google-10000-english`, 50-word list documents plus
  spelling documents (`apple: a, p, p, l, e.`)
- **Sentences:** TinyStories V2 train, first 250 MB
- **Sentences:** AG News train

| Split | Documents | Tokens | Raw bytes |
|---|---:|---:|---:|
| train | 432,781 | 69,966,259 | 280,266,715 |
| valid | 4,401 | 709,427 | 2,840,623 |
| test | 4,377 | 708,227 | 2,836,436 |

Total: 441,220 documents, 287,016,483 raw bytes.

## Training

- Fresh weights, batch 32, ctx 512, LR 4e-4, WSD over 3,300 steps
- bf16 autocast, dense MoE, no gradient checkpointing, 8 threads
- 3,600-second cap

Result: **3,300 steps / 54,067,200 tokens / 2,936.5 s** (48m 56s),
status `completed`, **0 non-finite skips**, ~18.4k tokens/s.
The step schedule finished before the 3,600-second cap.

## Final evaluation

| Eval | Tokens | BPB | Top-1 |
|---|---:|---:|---:|
| In-domain mixed test | 708,226 | 0.0661 | 95.27% |
| Official WikiText-2 test | 293,652 | **1.5355** | 52.74% |
| Internal WikiText-2 held-out split | 24,024 | 1.0951 | 61.36% |
| Official WikiText-103 test | 282,936 | **0.8551** | 64.00% |
| OOD Tiny Shakespeare | 337,118 | **1.1198** | 55.52% |

## What this means

- **WikiText-103 zero-shot is under 1.0 BPB** (0.8551).  This is the strong
  number, but WikiText-103 is still Wikipedia and the tokenizer is
  WikiText-2-derived, so it is not a strict out-of-distribution result.
- **Tiny Shakespeare — the genuinely different domain — is 1.1198 BPB**,
  close to but **not under 1.0**.  It improved from 1.5252 at step 2,000.
- Official WikiText-2 test is 1.5355 BPB; the model never saw WikiText-2
  text in training.
- In-domain 0.0661 BPB is severe fitting to the simple TinyStories/wordlist
  corpus and is not a quality claim.

Direct overlap check against 20M tokens of the non-WikiText training stream:
**0/293,622** WikiText-2 official 32-token shingles, **0/282,906**
WikiText-103 32-token shingles, **0/337,088** Shakespeare 32-token shingles.
No direct text leakage.

## Caveat and leaderboard status

The **tokenizer** is WikiText-2-derived.  A strict tokenizer-level zero-shot
claim would require relearning the subword vocabulary on the non-WikiText
corpus and retraining.  This is a model-weight zero-shot result.

This is **not** a WikiText leaderboard entry: the standard WikiText-2/103
benchmark trains on the corresponding train split.  It is a cross-dataset
zero-shot result.  A 3,300-step run also cannot support a leaderboard claim.

## Reproduction

```sh
python z3/train_v3.py \
  --device cuda --amp --amp-dtype auto \
  --preset kaggle10m_nomtp \
  --corpus data/nonwiki_bins/train.bin \
  --val-corpus data/nonwiki_bins/valid.bin \
  --data-manifest data/nonwiki_bins/manifest.json \
  --batch-size 32 --ctx 512 --lr 4e-4 \
  --steps 3300 --lr-total-steps 3300 \
  --eval-every 500 --eval-batches 1 \
  --max-seconds 3600 --no-grad-checkpoint --threads 8 \
  --moe-dense-dispatch \
  --out runs/hour_nonwiki
```

Zero-shot eval:

```sh
python z3/eval_domains.py \
  --checkpoint runs/hour_nonwiki/best.pt \
  --test-dir data/wikitext2_official_test \
  --tokenizer layered --document-framing --sidecar --require-sidecar \
  --ctx 512 --batch-size 16 --device cuda \
  --out data/final_wikitext2_official.json
```

Checkpoints and JSONs live in `/marimo/z4_smoke/`; they are not committed.
