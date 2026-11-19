# Balanced-corpus 20-minute continuation (2026-11-19)

This is the fallback run after the 7-hour attempts were lost/canceled. It
resumes the **40-minute broad20 checkpoint** (step 1400) and trains **700
more steps** on the 955M-token balanced corpus: 22,937,600 additional
tokens, 1,127.2 s, batch 64, ctx 512, LR 4e-4 WSD with decay over the
last 10%. Total training across the 40-minute and this run is **68.8M
tokens**.

## Metrics

| Eval | Tokens | BPB | Top-1 | Top-5 |
|---|---:|---:|---:|---:|
| Balanced in-domain valid | 12,154,535 | **0.2357** | **85.05%** | — |
| PTB test (clean OOD) | 105,105 | **0.9789** | **47.19%** | **65.30%** |

The in-domain number is optimistic: valid is a chunk-level split of the
same sources, so neighbouring chunks can leak. PTB is the honest number:
**0.9789 BPB / 47.19% top-1** at 68.8M total tokens.

For comparison, the lost 55-minute run reached 0.8679 BPB on the same PTB
file with 68.8M tokens but a different corpus. The balanced corpus is
slightly worse on PTB at the same token count, likely because it is more
math/code-heavy and less news-heavy.

Training loss fell from 1.93 at resume to ~0.75 by step 2090. Generation
is still word salad at 68.8M tokens, but stays in-domain for a sentence
(finance/legal/recipe prompts) and produces LaTeX-like math for the Python
prompt.

## 115M → 84M parity

The 115M `kaggle10m_nomtp` model carries 30,783,125 MTP parameters that
are never executed. The true no-MTP model is `kaggle10m_small` at
**84,455,112 parameters**.

Loading the 115M trunk into the 84M model (`strict=False`) gives:

| Check | Result |
|---|---|
| Missing trunk keys | **0** |
| Unexpected keys | 116 (MTP blocks/adapters, `mtp_output`, `hd_now/pos/perm`) |
| Parameter count | 84,455,112 |
| Eval logits max/mean diff | **0.0 / 0.0** |
| Eval hidden max/mean diff | **0.0 / 0.0** |
| Eval CE diff | **0.0** |
| Argmax agreement | **100%** |
| Shared gradients (dropout off) | **244/244 exactly 0.0** |

The only difference in training mode is dropout RNG: removing MTP modules
changes the RNG consumption order, so masks differ. With dropout disabled,
the graph is bit-identical.

**Conclusion:** the 84M model is a free drop-in for the 115M `nomtp`
trunk. It is not a capacity change; the active trunk is the same. It saves
~124 MB of weights (final.pt 509.7 MB → 337.9 MB) and removes the
misleading 115M parameter count. Making the model 115M again without MTP
would require adding ~30.8M parameters elsewhere (wider/deeper trunk),
which is a new architecture and not bit-compatible with this checkpoint.

## Persistence

- Kaggle: `shem42/z4-broad20m-balanced`
  - `final.pt` (115M), `best.pt`, `final84.pt` (84M), `config.json`,
    `result.json`, `train.log`, `eval_ptb.json`, `generation.log`
- Corpus: `shem42/z4-broad7h-bins` (955M tokens + tokenizer)

The 7-hour run was canceled; there was no 8-hour run. The 04:58 7-hour
attempt died after ~56 minutes and never uploaded a checkpoint, so those
metrics are unrecoverable.

## Next decision

Stay on the **84M no-MTP** model: identical active compute, identical
trunk metrics, smaller checkpoints. If a 115M model is required later,
widen the trunk (e.g., d_model or depth) rather than re-adding dead MTP
parameters. The 68.8M-token checkpoint is the best starting point for a
longer balanced-corpus run.
