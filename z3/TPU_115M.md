# Z4 115M on TPU/XLA

The trainer now has a maintained TPU path:

```sh
python z3/train_v3.py --preset kaggle10m_nomtp \
  --corpus data/wikitext_bins/train.bin \
  --val-corpus data/wikitext_bins/valid.bin \
  --data-manifest data/wikitext_bins/manifest.json \
  --device xla --amp --batch-size 4 \
  --out runs/z3_115m_tpu
```

`--device tpu` is an alias for `--device xla`.  `--device auto` now detects a
TPU environment (`TPU_NAME`, `COLAB_TPU_ADDR`, or `PJRT_DEVICE=TPU`) after
checking CUDA.

## What changed

- **bf16 XLA autocast.**  CUDA keeps fp16 AMP; XLA uses
  `torch_xla.amp.autocast` (falling back to
  `torch.autocast("xla", dtype=torch.bfloat16)`).  The CUDA `GradScaler` is
  not created on XLA.
- **XLA optimizer path.**  `xm.optimizer_step(optimizer)` plus
  `torch_xla.sync()` (falling back to `xm.mark_step()`).
- **Device selection.**  `xla`/`tpu` are accepted directly; `auto` recognizes
  a TPU environment.
- **Checkpoint portability.**  Existing `_to_cpu` / `_save_cpu` already move
  XLA tensors to CPU before `torch.save`; loading uses `map_location="cpu"`.
- **RNG state.**  XLA RNG state is captured/restored when
  `torch_xla.core.xla_model.get_rng_state` is available.
- **Evaluation.**  `eval_v3._resolve_device` accepts `xla`/`tpu` and
  `auto` detects TPU, so `eval_domains.py --device xla` works.
- **Model paths.**  Attention already switches to the manual causal path on
  `x.device.type == "xla"`; the MoE router already uses its fixed-shape dense
  dispatch on XLA.

## Multi-core

`z3/train_tpu.py` is an experimental 8-replica wrapper:

```sh
python z3/train_tpu.py --nprocs 8 --out runs/z3_115m_tpu \
  --preset kaggle10m_nomtp \
  --corpus data/wikitext_bins/train.bin \
  --val-corpus data/wikitext_bins/valid.bin \
  --data-manifest data/wikitext_bins/manifest.json \
  --batch-size 4
```

It uses `torch_xla.distributed.xla_multiprocessing.spawn`, gives each rank a
different seed, writes only the master checkpoint to `--out`, and deletes the
rank-local `/tmp/z3_tpu_rank*` directories.  Gradient reduction is delegated
to `xm.optimizer_step`, so all replicas must stay in lockstep.  This wrapper
has **not** been validated on a TPU from this environment; start with the
single-device path.

## Kaggle notebook

[`kaggle_tpu_115m.ipynb`](kaggle_tpu_115m.ipynb) checks the TPU, installs
`torch_xla` only if missing, builds WikiText-2 bins once, then runs the
measured `kaggle10m_nomtp` profile with `TPU_MODE = "single"` or `"multi"`.
It retries the single-device path if the multi-core launch fails.

## Caveats

- XLA compilation is counted inside the 600-second cap; the first step is
  slower than steady state.
- Single-device TPU uses one core of a v3-8/v5e-8 slice.  The 8-replica
  wrapper is the performance path.
- TPU bf16 is not bit-comparable to CUDA fp16 or CPU fp32.  Compare BPB at
  equal wall time on the same corpus, not step counts.
- No TPU result is claimed yet.  The measured best remains the T4
  `kaggle10m_nomtp` result (1.5981 test BPB / 1.6425 validation BPB).
