# Kaggle large-input mount incident and workarounds (2026-11-22)

## Symptom

GPU kernels that mounted the 12h weights dataset or the 2 GB corpus dataset
stayed `QUEUED` for hours, while CPU kernels using the same datasets started
immediately. The GPU12h training run itself had completed normally; no
checkpoint was lost.

## Controlled tests (all on `NvidiaTeslaT4`)

| Input | Behaviour |
|---|---|
| No dataset (`z4-gpu-queue-probe`) | COMPLETE in 11 s, Tesla T4 |
| 19-byte private dataset (`z4-tiny-private-probe`) | COMPLETE |
| 3 KB public dataset (`z4-fineweb-external-test`) | COMPLETE |
| 338 MB kernel output (`z4-causal-gpu12h/final.pt`) | COMPLETE in ~12 s |
| 338 MB private dataset (`z4-causal-gpu6h-weights`) | QUEUED 5+ min |
| 646 MB private dataset (`z4-causal-gpu12h-weights`) | QUEUED 4+ h |
| 2 GB private dataset (`z4-broad7h-bins`) | QUEUED 4+ h |
| 1.91 GB kernel output (`z4-corpus-mirror`) | QUEUED 5+ min |
| 324 MB kernel output (`z4-corpus-150m-mirror`) | COMPLETE in <45 s |

Conclusion: GPU capacity and the probe were not the problem. **Large input
mounts (roughly >350 MB) are stuck on GPU workers.** Small datasets and small
kernel outputs mount normally. Dataset privacy is not the discriminator: a
19-byte private dataset worked, a 338 MB private dataset did not.

There is also an account-level **maximum of 2 batch GPU sessions**. Queued
kernels count toward that limit, which is why new GPU pushes were rejected
with `Maximum batch GPU session count of 2 reached.` The redundant GPU domain
eval was deleted to free a slot.

## Workarounds

1. **Mount checkpoints as kernel output**, not as a dataset:
   `kernel_sources: ["shem42/z4-causal-gpu12h"]` exposes
   `/kaggle/input/notebooks/shem42/z4-causal-gpu12h/runs/causal_gpu12h/final.pt`.
2. **Mirror the corpus as a bounded subset.** `z4-corpus-150m-mirror` copies
   150,000,000 tokens (300,000,000 bytes) sampled in 30 evenly spaced slices
   from the 955M-token corpus, plus `valid.bin` (24 MB) and the tokenizer.
   Total ~324 MB, mounts in under a minute. 150M tokens is ~3× the 6h token
   budget, so one block does not repeat data.
3. **Generate a corpus at runtime** for short probes (repo text + layered
   tokenizer), so the probe needs no dataset at all.
4. **Use CPU for large-dataset work.** CPU kernels mount the full 2 GB corpus
   and the weights datasets without delay; the CPU per-domain eval and the
   corpus mirrors use this.

## AMP result

The self-contained probe (`z4-gpu-amp-probe2`, checkpoint via kernel output,
corpus generated from repo text) completed:

| AMP | tok/s | non-finite skips |
|---|---:|---:|
| bf16 | 2,530 | 0 |
| **fp16** | **4,669** | **0** |

fp16 is **1.85×** faster on the T4, matching the trainer's own `auto` policy
(fp16 for compute capability < 8.0). A 6-hour block at this rate covers about
100.8M tokens instead of 51.6M.

## Consequence for training

The next 6-hour block (`z4-causal-gpu18h`) uses **zero dataset sources**:

- checkpoint via `shem42/z4-causal-gpu12h`
- 150M corpus via `shem42/z4-corpus-150m-mirror`
- fp16 AMP, lr 1e-4, `--decay-frac 0.5`

It should schedule like the 338 MB / 324 MB probes (seconds), not like the
large-dataset jobs.
