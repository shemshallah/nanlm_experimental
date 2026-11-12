# Colab: 30-minute hierarchical multidomain run

Open [`colab_multidomain_30min.ipynb`](colab_multidomain_30min.ipynb) in Google
Colab, choose **Runtime → Change runtime type → GPU**, then **Run all**.

No Kaggle dataset or private credential is needed. The notebook clones this
repo and builds the corpus from the public NaNROUTER `corpora/domains/`
releases.

## What the run does

1. Samples **4 MB from each of 16 domains** with HTTP range requests
   (`wikipedia-en, arxiv, pubmed, law, patents, code, math, news, recipes,
   security, qa, multilingual, finance, standards, humanities, chemistry`).
2. Builds 43.7M train / 4.7M valid byte positions, document-level 90/10 split.
3. Trains a **151M-total / ~21M-active** additive MoE with:
   - domain token per window
   - **hierarchical routing: 8 branches × 16 pairs, choose one branch, top-2
     pairs inside it**
   - linear+geometry leaf scores, learnable geometry strength
   - balance loss 0.01
4. Stops training at **30 minutes** (`timeout 1800`), then probes 16 domains
   (131,072 held-out positions) with `probe_r32.py`.

## Expected throughput

Measured on Kaggle 2× T4 with the same 151M additive MoE and `grouped` dispatch:

- ~4.3k tokens/s at batch 32, context 256
- 30 minutes ≈ **7.7M positions** (~940 steps), not 10M
- A100/H100 will do substantially more in the same wall time

The notebook uses `--batch 32`. If you hit CUDA OOM on a smaller GPU, change to
`--batch 16` (≈2.5k tokens/s, so ~4.5M positions in 30 minutes).

## Colab limits (2026)

There is **no published weekly Colab GPU allowance**.

| Plan | Compute | Approx. GPU time | Notes |
|---|---|---|---|
| Free | dynamic, no units | not guaranteed; max 12 h session | often T4; can end early; no SLA |
| Pro | 100 units/month | ~55–85 T4-h or ~18 A100-h (measured rates) | T4/L4/V100/A100 when available |
| Pro+ | 500–600 units/month* | priority A100/H100; up to 24 h with browser closed | background execution |
| Pay as you go | units | same rates | units expire |

*Sources differ on Pro+ units (500 vs 600); Google does not publish per-GPU
hour rates. Treat these as observed ranges, not entitlements.

**There is no T100 GPU** in Colab or Kaggle. Colab offers T4, L4, V100, A100
(40/80GB), G4, and H100 depending on plan/availability. Kaggle offers T4×2 or
P100 with a **30 GPU-hour/week** quota.

Colab Enterprise on Google Cloud is pay-as-you-go, e.g. list prices around
$0.42/h T4, $4.71/h A100 40GB, $11.75/h H100 (spot/committed lower). It needs
GCP billing and enabled APIs; a service-account JSON alone is not enough.

The Google credential in `Documents/apikeys/google_nanlm.json` is a GCP
service account for project `nanlm1`. It can mint access tokens, but:
- Colab consumer has no public API to launch GPU runtimes;
- the project has `aiplatform`, `cloudbilling`, and `cloudresourcemanager` APIs
  disabled, so Colab Enterprise/Vertex AI cannot be launched as-is;
- enabling those APIs and creating runtimes would be billable GCP usage.

For a 30-minute test, free Colab or Kaggle is enough. Use Colab Pro/Pro+ only
if you specifically need A100/H100 availability.

## Sending results back

The last notebook cell downloads `r32_results.zip` containing
`hier.probe.json` and the training log. Send that back and I’ll log the run in
`r32/EXPERIMENTS.md` and compare it with the flat-routing 16-domain result.

## If you want a plain baseline

Run the same notebook a second time with the `plain21` command from
`r32/kernels/pilot/train.py` (dim 576, no MoE) and probe it the same way. The
notebook’s 30-minute window is for one arm on one GPU.
