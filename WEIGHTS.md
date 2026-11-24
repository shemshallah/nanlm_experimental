# Z4 causal 84M checkpoints

Public weights for the causal 84M Z4 lineage: **84,455,112 parameters**, no
MTP, `causal_concept=True`, layered tokenizer (64,060 vocab). Every number
below is on the non-WikiText corpus; **PTB is the clean OOD benchmark**.

| Release | Total tokens | PTB BPB | PTB top-1 | Valid BPB | Valid top-1 | Kaggle mirror |
|---|---:|---:|---:|---:|---:|---|
| [124M tokens (6h)](https://github.com/shemshallah/nanlm_experimental/releases/tag/z4-causal-84m-124m-tokens) | 124,179,072 | 1.7624 | 28.48% | 1.2839 | 35.81% | [dataset](https://www.kaggle.com/datasets/shem42/z4-causal-gpu6h-weights) |
| [176M tokens (12h)](https://github.com/shemshallah/nanlm_experimental/releases/tag/z4-causal-84m-176m-tokens) | 175,764,096 | 1.7349 | 29.04% | 1.2488 | 36.86% | [dataset](https://www.kaggle.com/datasets/shem42/z4-causal-gpu12h-weights) |
| [269M tokens (18h)](https://github.com/shemshallah/nanlm_experimental/releases/tag/z4-causal-84m-269m-tokens) | 269,193,856 | **1.6849** | **30.01%** | **1.2030** | **38.14%** | [dataset](https://www.kaggle.com/datasets/shem42/z4-causal-gpu18h-weights) |

## Assets

Each release contains:

- `<prefix>-final.pt` — end-of-block checkpoint; PTB numbers above are for this file
- `<prefix>-best.pt` — best valid checkpoint (selected on a 4,096-token valid batch)
- `<prefix>-config.json`
- `<prefix>-result.json`
- `<prefix>-eval-ptb.json`
- `<prefix>-train.log`

The files are standard PyTorch checkpoints compatible with `z3/eval_v3.py`,
`z3/generate_v3.py`, and `z3/train_v3.py --resume`.

## Generate

```sh
git clone https://github.com/shemshallah/nanlm_experimental.git
cd nanlm_experimental
python z3/generate_v3.py \
  --checkpoint /path/to/z4-causal-84m-269m-final.pt \
  --layered-model z3/tokenizers/layered_v1/layered_tokenizer.json \
  --device cpu \
  --max-new-tokens 60 \
  --prompt "The capital of France is"
```

## Resume training

```sh
python z3/train_v3.py ... --resume /path/to/z4-causal-84m-269m-final.pt
```

Use the same preset/architecture flags as the checkpoint
(`--preset kaggle10m_small --causal-concept --no-ema`, batch/ctx as desired).
The optimizer state is included.

## Honest status

The model is **not competitive**. At 269M tokens / 84M parameters it has seen
only 3.2 tokens per parameter, far below compute-optimal training. Generation
is locally fluent but globally incoherent, and PTB 1.6849 is weak for an 84M
model. These releases are a reproducible checkpoint lineage, not a claim of
frontier quality.

## Publishing a new checkpoint

The workflow `.github/workflows/publish-weights.yml` republishes a kernel
output as a GitHub Release. It downloads directly from Kaggle using the
`KAGGLE_API_TOKEN` repository secret and uploads the same six assets.

License: **CC0-1.0**.
