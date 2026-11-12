# z2 — 30-minute Colab run

Open the notebook directly:

https://colab.research.google.com/github/shemshallah/nanlm_experimental/blob/master/z2/z2_colab.ipynb

## Before running

1. `Runtime` → `Change runtime type` → **T4 GPU**.
2. Run the cells in order.
3. Mount Google Drive when prompted. Artifacts go to
   `/content/drive/MyDrive/nanlmv3_z2`.

Do not run this on CPU. It will be far too slow and the 30-minute cap will be
wasted.

## What it trains

The 84,859,523-parameter v2 hybrid:

- RoPE, Pre-LN, RMSNorm, SwiGLU
- 8 layers, 8 heads, `d_model=256`, `ctx=512`
- two 128-pair mirror MoE layers, top-8
- 32 learnable hyperbolic branches
- 1M-entry hashed 4-gram memory
- EMA weights, time-cosine LR to zero

Corpus, mixed to 200MB:

- `corpora-paragraphs-0010`
- `corpora-short-stories-0004`
- `corpora-sentences-0010`
- `hw5-gutenberg-c256-g1` (21-book Gutenberg reference)

Tokenizer: the z1 32k BPE tokenizer. If it is missing, the script trains a new
32k BPE on a 50MB slice.

## Runtime

- Download + extract + corpus build + encode: ~4–6 min
- Training cap: **1,800 s / 30 min**
- Total Colab wall time: ~35–40 min

## Outputs

```text
/content/drive/MyDrive/nanlmv3_z2/
  best.pt
  final.pt
  ema.pt
  result.json
  history.json
  config.json
```

`result.json` contains best/final/EMA metrics, elapsed time, parameter count,
and the geometry state.

## Resume from `best.pt`

The run cell now passes `--resume-best`. The script automatically checks:

```text
/content/drive/MyDrive/nanlmv3_z2/best.pt
```

If that file exists, it is loaded with `--init-from` and evaluated first
(`--eval-before`), then training continues.

Useful variants:

```python
# fresh run, ignore an existing best.pt
!python .../train_colab.py --minutes 30 --out "$OUT" --no-resume-best

# specific checkpoint file
!python .../train_colab.py --minutes 30 --out "$OUT" \
  --checkpoint-path /content/drive/MyDrive/nanlmv3_z2/best.pt

# remote checkpoint
!python .../train_colab.py --minutes 30 --out "$OUT" \
  --checkpoint-url "https://github.com/.../best.pt"
```

Notes:

- `best.pt` is written after the first validation eval. If the job died before
  any eval, there is no checkpoint to recover.
- The checkpoint must be the same 84.86M v2 architecture.
- If a checkpoint is corrupt or from another configuration, delete/rename it
  or rerun with `--no-resume-best`.

## Optional continuation

Pass a checkpoint URL to start from an existing build:

```python
CHECKPOINT_URL = "https://github.com/.../best.pt"
!python /content/nanlm_experimental/z2/train_colab.py \
  --minutes 30 --out "$OUT" --checkpoint-url "$CHECKPOINT_URL"
```

Without it, z2 starts from fresh weights.

## Notes

- The script checks the corpus release SHA256SUMS.
- Existing files are reused on rerun, so a second run does not re-download.
- If Colab disconnects, the Drive artifacts remain. Reopen the notebook and
  continue from the saved checkpoint with `--checkpoint-url` pointing at a
  downloadable copy.
- No Kaggle API token is required.
