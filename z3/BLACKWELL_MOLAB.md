# Z4 115M on Blackwell / Molab

This is the CUDA/Blackwell port of the measured `kaggle10m_nomtp` profile.
It keeps the 115,238,237-parameter model and the 600-second WikiText-2
protocol.  It is not the Kaggle TPU path.

## Files

- [`blackwell_marimo.py`](blackwell_marimo.py) — marimo app for Molab
- [`blackwell_115m.ipynb`](blackwell_115m.ipynb) — plain Jupyter/Colab notebook
- [`train_v3.py`](train_v3.py) — `--amp-dtype auto|fp16|bf16`
- [`BLACKWELL_MOLAB.md`](BLACKWELL_MOLAB.md) — this file

## Measured 2026-11-18

- 600-second cap: 1,100 steps, 9,011,200 tokens, **0.5336 validation BPB**
  and **0.5183 held-out WikiText-2 split BPB** (71.55% top-1).
- OOD Tiny Shakespeare: **4.4842 BPB** (8.50% top-1).  The model is a narrow
  Wikipedia/WikiText-2 fit, not a general language model.
- Official WikiText-103 test transfer: **0.7905 BPB** (59.63% top-1).  This
  is same-domain Wikipedia transfer, **not** a WikiText-103 benchmark or
  leaderboard claim.

See [`BLACKWELL_600S_2026_11_18.md`](BLACKWELL_600S_2026_11_18.md) for the
full tables, overlap checks, and reproduction commands.

## Mixed word-list / sentence probe

A fresh 300-second run on shuffled word lists + TinyStories + WikiText-103
(4,603,904 tokens, 0.57 epoch) reached:

- mixed held-out test: **1.6313 BPB**
- OOD Tiny Shakespeare: **3.1724 BPB** (down from 4.4842; below the ~3.59
  uniform baseline)
- official WikiText-103 test transfer: 2.0843 BPB

This is a real generalization improvement, but not under 1.0 BPB.  See
[`MIXED_300S_2026_11_18.md`](MIXED_300S_2026_11_18.md).

## Non-WikiText 1-hour zero-shot

Fresh weights trained only on word lists + TinyStories + AG News
(54,067,200 tokens, no WikiText text):

- Official WikiText-2 test: **1.5355 BPB** (52.74% top-1)
- Official WikiText-103 test: **0.8551 BPB** (64.00% top-1)
- OOD Tiny Shakespeare: **1.1198 BPB** (55.52% top-1)

Shakespeare is the strict OOD check and is close to, but not under, 1.0.
The subword tokenizer is WikiText-2-derived, so this is model-weight
zero-shot, not strict tokenizer-level zero-shot.  See
[`NONWIKI_HOUR_2026_11_18.md`](NONWIKI_HOUR_2026_11_18.md).

## Requirements

- A CUDA GPU.  Blackwell (`sm_100`/`sm_120`) needs a PyTorch build with
  Blackwell kernels; install a current CUDA 12.8/13.x wheel (for example
  `torch==2.9.0` from the matching PyTorch index).
- `git`, internet access for the first WikiText-2 download, and enough disk
  for the repo plus the 115M checkpoints.
- `tokenizers` and `pyarrow` (the app installs them if missing).
- `nvidia-smi` available.

Check the environment first:

```sh
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
nvidia-smi
```

## Molab

Upload [`blackwell_marimo.py`](blackwell_marimo.py) to Molab, or open it
from the GitHub repo.  The app:

1. Prints the Python/torch/CUDA/GPU environment.
2. Clones the repo if it is not already present.
3. Downloads WikiText-2 and builds the token bins once.
4. Runs `train_v3.py --device cuda --amp --amp-dtype auto` with the selected
   batch size, context, and wall cap.
5. Runs the held-out test evaluation and shows a summary table.

The default controls match the T4 profile: `kaggle10m_nomtp`, batch 4,
ctx 512, 600 seconds, gradient checkpointing disabled.  Re-enable
checkpointing only if the GPU is memory-constrained.

## Command line

```sh
python z3/train_v3.py \
  --device cuda --amp --amp-dtype auto \
  --preset kaggle10m_nomtp \
  --corpus data/wikitext_bins/train.bin \
  --val-corpus data/wikitext_bins/valid.bin \
  --data-manifest data/wikitext_bins/manifest.json \
  --batch-size 4 --ctx 512 --max-seconds 600 \
  --no-grad-checkpoint \
  --out runs/z4_115m_blackwell
```

Then:

```sh
python z3/eval_domains.py \
  --checkpoint runs/z4_115m_blackwell/best.pt \
  --test-dir data/wikitext_bins/test \
  --data-manifest data/wikitext_bins/manifest.json \
  --tokenizer layered --document-framing --sidecar --require-sidecar \
  --ctx 512 --batch-size 4 --device cuda \
  --out runs/z4_115m_blackwell/test_domains.json
```

## AMP behavior

`--amp-dtype auto` selects:

- **bf16** on CUDA compute capability >= 8.0 (A100, H100, Blackwell)
- **fp16** on T4/V100
- bf16 on XLA/TPU

The CUDA `GradScaler` is only created for fp16.  bf16 does not need loss
scaling, so the optimizer takes the plain `loss.backward()` path.

## Comparison caveat

The measured T4 best is **1.5981 test BPB / 1.6425 validation BPB** after
900 steps and 1,843,200 tokens.  A Blackwell run is a new hardware result;
compare at the same wall time, corpus, and tokenizer, not by step count.
No Blackwell result is claimed yet.
