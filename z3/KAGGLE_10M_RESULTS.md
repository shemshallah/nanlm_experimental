# Z4 115M Kaggle 10-minute result

This is the first real 10-minute 115M run on a Kaggle T4 using the
`kaggle10m` short-run profile.  It is an early-training comparison, not a
frontier claim.

## Run

- Kaggle kernel: <https://www.kaggle.com/code/shem42/z4-115m-kaggle-10m>
- Notebook: [`kaggle_10m_115m.ipynb`](kaggle_10m_115m.ipynb)
- Preset: `kaggle10m`
- GPU: Tesla T4, 14.6 GiB
- Data: WikiText-2 raw train+valid, 2,566,566 train tokens and 24,207 valid
  tokens; 24,024 held-out test tokens; 4.4922 bytes/token
- Context / batch: 512 / 4
- Wall cap: 600 seconds

## Training

| Metric | Value |
|---|---:|
| Parameters | 115,238,237 |
| Steps completed | 599 |
| Tokens seen | 1,226,752 |
| Training wall | 604.1 s |
| Non-finite skips | 0 |
| Warmup steps | 30 |
| Best eval | step 599 |
| Best validation BPB | **1.8254** |
| Final validation BPB | **1.8254** |
| Best validation top-1 / top-5 | 24.19% / 38.09% |
| Held-out test BPB | **1.7914** |
| Held-out test top-1 | 25.93% |

The validation BPB uses the exact target-byte denominator from the aligned
sidecar.  The test set is the builder's held-out WikiText-2 split.

## Feedback and object gates

At the end of the run:

| Gate | Value |
|---|---:|
| Token feedback | **-0.03750** |
| Route-hidden feedback | -0.00754 |
| Route-concept feedback | -0.00884 |
| Memory feedback | +0.00656 |
| MTP-carry feedback | +0.00083 |
| Object-depth carry | -1.02e-06 |
| Object gate | **0.01943** |
| Feedback progress | 1.0 |

The token and route feedback paths opened.  The object gate is still
**0.01943 < 0.05**, so the object memory is effectively closed after 599
steps; that remains the main short-run dead path.

## Comparison caveats

- The 10-minute run saw only 1.23M tokens.  It is undertrained.
- This is WikiText-2, not FineWeb-Edu.  Do not compare it directly to the
  Blackwell FineWeb 10-minute numbers (1.944 BPB byte model / 2.663 BPB
  subword WikiText-103) or to the synthetic CI smoke.
- The `kaggle10m` profile skips the two deep-supervision full-vocab heads
  (20.5% of estimated FLOPs) and the full-byte sequence probe (4.9%), lowers
  the MTP weight to 0.1, uses a 30-step warmup, and disables EMA.  The result
  is the combined profile, not an isolated ablation of any one change.
- A stock `115m` same-hardware baseline is the next control.  Only an
  equal-wall comparison can attribute the gain.

## First attempt

The first Kaggle attempt was stopped after its data-prep fallback converted a
2.05 GB FineWeb-Edu parquet to 3.47 GB of text and then tried to build bins
from the whole file.  The public Kaggle range URL returned 404 inside the
kernel.  The fixed notebook uses WikiText-2 for the fast gate; FineWeb should
be pre-sliced or pre-tokenized before a repeat run.
