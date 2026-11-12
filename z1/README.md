# z1 — nanlmv3 v2 new-corpus build

Kaggle kernel: https://www.kaggle.com/code/shem42/nanlmv3-v2-newcorpus10

This directory contains the **build metadata and training history** for the
10-minute v2 run on the newer NaNROUTER corpus. Model checkpoints are not
committed, matching this repo's `*.pt` policy; they remain in the Kaggle kernel
output.

## Corpus

- `corpora-paragraphs-0010` (published 2026-11-10)
- `corpora-short-stories-0004` (published 2026-11-10)

100MB combined corpus, BPE 32k trained on the new corpus.

## Build

- 84,859,523 params
- ctx 512, batch 8, 8 layers, 8 heads, SwiGLU 1024
- 128 mirror-pair MoE experts, top-8, two MoE layers
- 32 learnable hyperbolic branches
- 1M-entry hashed 4-gram memory
- 4,414 steps in 612 s (600 s cap)
- LR 1e-4, time-cosine to zero, EMA 0.999 from 30%
- 1,024-token fixed validation

## Measured metrics

| Checkpoint | Bits/token | Top-1 | Top-5 |
|---|---:|---:|---:|
| Best model, step 4,000 | 10.4644 | 6.84% | 19.34% |
| **Final model, 4,414** | **10.4484** | **7.03%** | 19.14% |
| Final EMA | 10.4584 | 6.84% | 19.14% |

Final beats best on bits/token, so the end-regression fix holds on the new
corpus. The curve is still descending at the 10-minute cutoff.

## Files

```text
z1/
  README.md
  build/
    result.json      # best/final/EMA metrics, artifacts, geometry state
    history.json     # per-eval model and EMA curves
    config.json      # full run configuration
    train.log        # kernel log converted to text
    MANIFEST.json    # provenance, checkpoint paths, small-file checksums
  code/
    unified_v2.py    # exact model/training code for this build
  kernels/
    newcorpus10/
      train.py
      kernel-metadata.json
```

## Continuation — 10 more minutes on new data

Kernel: https://www.kaggle.com/code/shem42/nanlmv3-v2-continue10

Loaded this z1 `best.pt`, evaluated it on a new held-out slice from
`corpora-paragraphs-0009` + `corpora-short-stories-0003`, then trained 600 s
at LR 5e-5.

| Checkpoint | Bits/token | Top-1 | Top-5 |
|---|---:|---:|---:|
| Baseline (z1 on new val) | 10.3721 | 8.59% | 20.70% |
| Best after continuation | **10.0394** | 10.64% | 22.46% |
| Final after continuation | 10.0442 | **10.94%** | **22.66%** |
| Final EMA | 10.0467 | 10.64% | 22.17% |

Improvement: **-0.333 bits/token, +2.35 pp top-1**. Final is effectively tied
with best (0.005 bits), with higher top-1/top-5. The continuation rewards are
in `build/continuation10_result.json` and `build/continuation10_history.json`.

## Checkpoints

The Kaggle kernel output contains:

```text
build/v2_newcorpus10/best.pt
build/v2_newcorpus10/final.pt
build/v2_newcorpus10/ema.pt
```

All three are ~340 MB and are intentionally not committed.
