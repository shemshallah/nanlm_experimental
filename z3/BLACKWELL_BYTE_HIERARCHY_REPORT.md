# Blackwell byte hierarchy — 64 experts, batch 64

This is the byte-model follow-up with:

- byte vocabulary 273, byte-level top-1/top-5
- 64 routed experts (up from 32)
- batch 64 × 1,024-byte context
- GPU-resident byte pool; no host-side random-window sampler in the hot path
- byte → subword → word/sentence feedback:
  - byte-aligned subword IDs
  - subword embedding in the concept encoder
  - word/sentence pools from byte-aligned spans
  - concept/object state fed into the MoE router
- 64-expert, 238,177,795-parameter model
- activation checkpointing
- reduced concept candidates (4 vertices / 2 pairs / 1 triple) to fit memory

## Result

| Metric | Value |
|---|---:|
| Steps | **119** |
| Wall | 604.3s |
| Bytes seen | 7,798,784 |
| Step time | 5.08s |
| BPB valid | **2.635735** |
| Top-1 valid | **48.91%** |
| Top-5 valid | **77.33%** |
| Params | 238,177,795 |
| GPU peak | ~71.7 GB / 96 GB |
| Non-finite | 0 |

The requested 500-step milestone was **not reached**: the hierarchical 238M
model at batch 64 is 5.08s/step, so the 600s cap stopped at 119 steps. The
logger is configured for 500; there is no fabricated 500-step line. To get
500 steps in 10 minutes this configuration must be ~4.2× faster or run longer.

## What fit

- Batch 64 did fit after reducing concept candidates and enabling checkpointing.
- 64 experts fit.
- The 4MB GPU-resident pool and byte-aligned sidecar removed the old host
  sampler from the hot path.
- A 22GB stale CUDA cache from the earlier session was cleared before launch.

## Artifacts

In the Molab session:

- `/marimo/runs/byte_hib_10m/final.pt`
- `/marimo/runs/byte_hib_10m/result.json`
- `/marimo/runs/byte_hib_10m/train.log`

The earlier exact 112.9M byte parity run remains:

- `/marimo/runs/old_byte_115m/final.pt`
- 717 steps / 23.5M bytes / 2.1945 FineWeb BPB / 55.69% top-1 / 84.26% top-5
