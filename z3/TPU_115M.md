# Z4 115M on TPU/XLA

## Status: code ready, TPU blocked by Kaggle account verification

No TPU result exists yet.  The 20-hour TPU quota is untouched (`0s` used).
Every pushed "TPU" kernel still runs on the standard CPU image
(`COLAB_IMAGE_TYPE=cpu`, no `/dev/vfio`), so `xm.get_xla_supported_devices()`
times out.

Two separate Kaggle issues had to be fixed to even request a TPU:

1. **Machine shape must be explicit.**  `enableTpu: true` alone is not enough.
   The saved kernel needs `machineShape: "TpuV5E8"` (or the legacy
   `"Tpu1VmV38"`, which Kaggle substitutes with the default TPU).
2. **A stale CPU image pin overrides the TPU VM.**  The kernel may carry
   `dockerImage: "gcr.io/kaggle-images/python@sha256:..."`, and the server then
   runs the TPU request in that CPU image.  Push with
   `dockerImage: ""` and `dockerImagePinningType: "latest"` to clear it.
   Verify with `GetKernel` that `machineShape` shows `TpuV5E8` and
   `dockerImage` is absent.

The `z4-115m-tpu` kernel now has the correct metadata.  With the image pin
cleared, a TPU push still lands on CPU and the quota stays at zero, which
points at Kaggle's one-time **Persona identity verification** gate for TPUs
(GPUs only require phone verification).  Complete it at
<https://www.kaggle.com/settings>; after that, rerun the notebook.  Do not
record a TPU result until the log shows a TPU device.

### Push recipe (Kaggle API)

Use the newer service endpoint; the legacy `/api/v1/kernels/push` silently
ignores `machineShape`.

```sh
POST https://api.kaggle.com/v1/kernels.KernelsApiService/SaveKernel
{
  "slug": "shem42/z4-115m-tpu",
  "machineShape": "TpuV5E8",
  "dockerImage": "",
  "dockerImagePinningType": "latest",
  "enableTpu": true,
  "kernelExecutionType": "QUICK_SAVE"
}
```

`dockerImage: ""` is the important part: omitting the field leaves the old
CPU image pin in place.  `GetKernel` must show `machineShape: "TpuV5E8"` and
no `dockerImage` before the run.

## First TPU run (2026-11-18)

- Kernel version 10, TPU v5e-8, 8 XLA devices (`xla:0` … `xla:7`)
- Image stack: `torch 2.8.0+cpu`, `torch_xla 2.8.0`, `/dev/vfio` present
- Parameters: 115,238,237 (`kaggle10m_nomtp`, MTP compute skipped)
- 600-second cap: 2 steps / 4,096 tokens; wall 1,265.9 s including the
  first-step XLA compile and the validation pass
- Validation after step 2: loss 11.1155, **3.6063 BPB** (2 batches,
  4,088 target bytes/tokens).  With the 30-step warmup the LR was still
  ~1e-5, so this is essentially the random-initialization baseline
  (`log2(64060) / 4.49 ≈ 3.56 BPB`), not evidence about the architecture.
- Test evaluation initially failed with
  `RuntimeError: Cannot set version_counter for inference tensor` because
  `torch.inference_mode()` produces inference tensors that the lazy XLA
  linear cannot materialize.  `eval_v3.py` now uses `torch.no_grad()` on
  XLA; the notebook also falls back to CPU for the test eval.
- Version 11 with the fix is queued.  The test BPB will be recorded only
  after that run completes.

The trainer has a maintained TPU path:

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
- **Fixed-shape segment pooling.**  Concept segment pooling used
  `torch.unique`, whose output size changes with every batch.  XLA treats that
  as a new graph and recompiles, so the TPU run stalls.  On XLA the trainer
  now uses a sort-based, fixed-size compaction (`_fixed_segment_ids`) and
  skips the unique-based object diagnostics.  Semantics are unchanged; only
  the diagnostic entropy/unique-ratio columns are reported as zero on XLA.

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

**Pin a matching torch/torch_xla pair.**  The Kaggle TPU image currently
ships **torch 2.11**, while the libtpu wheel index exposes `torch_xla`
2.8.0/2.8.1/2.9.0.  `torch_xla` 2.9.0 also pins the missing
`libtpu==0.0.21`; the index has `0.0.21.1`.  The notebook installs:

```sh
pip install torch==2.9.0 \
  --index-url https://download.pytorch.org/whl/cpu
pip install --no-deps torch_xla==2.9.0 \
  -f https://storage.googleapis.com/libtpu-releases/index.html
pip install libtpu==0.0.21.1 \
  -f https://storage.googleapis.com/libtpu-releases/index.html
```

and runs all training/eval in child processes, so the notebook never imports
the old torch 2.11 ABI.  The CPU wheel is used because the plain PyPI wheel
pulls several GB of CUDA dependencies that a TPU VM never uses; the libtorch
ABI is the same.  Do not install torch 2.11 with a 2.9 `torch_xla`: the
`_XLAC` import fails on removed `TensorImpl` symbols.

On TPU, pass `--no-grad-checkpoint`.  The 115M activations fit, and
recomputation makes the XLA graph much larger and slower to compile.

## Caveats

- XLA compilation is counted inside the 600-second cap; the first step is
  slower than steady state.
- Single-device TPU uses one core of a v3-8/v5e-8 slice.  The 8-replica
  wrapper is the performance path.
- TPU bf16 is not bit-comparable to CUDA fp16 or CPU fp32.  Compare BPB at
  equal wall time on the same corpus, not step counts.
- No TPU result is claimed yet.  The measured best remains the T4
  `kaggle10m_nomtp` result (1.5981 test BPB / 1.6425 validation BPB).
