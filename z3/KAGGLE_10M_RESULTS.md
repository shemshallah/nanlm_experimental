# Z4 115M Kaggle 10-minute result

This is the first real 10-minute 115M run on a Kaggle T4 using the
`kaggle10m` short-run profile.  It is an early-training comparison, not a
frontier claim.

**Current best short-run profile: `kaggle10m_lr3e4`, 1.7489 test BPB /
1.7835 validation BPB after 592 steps.**  The original 2e-4 result below is
kept as the historical reference.

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

## Equal-wall stock 115M baseline

The stock `115m` preset was run on the same WikiText-2 bins, same T4, same
600-second cap, with `--batch-size 4`.  It **OOM'd at batch 4** and the
notebook retried at batch 2.  The optimized profile fit batch 4 because it
skips the aux heads, the full-byte sequence probe, and the unused EMA copy.

| Run | Batch | Steps | Tokens | Best val BPB | Test BPB | Test top-1 |
|---|---:|---:|---:|---:|---:|---:|
| **`kaggle10m`** | **4** | 599 | **1,226,752** | **1.8254** | **1.7914** | **25.93%** |
| stock `115m` | 2 (4 OOM) | 682 | 698,368 | 1.9567 | 1.9206 | 21.50% |

The optimized profile processed **1.76x more tokens** in the same wall time
and is **0.129 BPB better** on the held-out test.  This is the combined effect
of batch 4 fitting (no wasted EMA/aux memory), the 30-step warmup, and the
reduced auxiliary loss weight; it is not a single-variable ablation.

### Baseline gates

The stock baseline used `feedback_gate_warmup_steps=500`, so its gates were
still ramping at the end: token **-0.02169**, route-concept **-0.01535**,
route-hidden **-0.00155**, memory **+0.01074**, MTP-carry **+0.00012**,
object-depth **-0.00021**, object gate **0.01918**.  The optimized run's
30-step warmup opened the token gate further (**-0.03750**) and kept the
object gate closed (**0.01943**).

## LR and batch sweep

Run: <https://www.kaggle.com/code/shem42/z4-115m-batch-lr-sweep>  
Same WikiText-2 bins, same 600-second cap.  `batch8` OOM'd and fell back to 6.

| Arm | Actual batch | LR | Steps | Tokens | Best val BPB | Test BPB | Test top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `kaggle10m` reference | 4 | 2e-4 | 599 | 1,226,752 | 1.8254 | 1.7914 | 25.93% |
| batch8 → 6 | 6 | 2e-4 | 442 | 1,357,824 | 1.8890 | 1.8200 | 24.84% |
| batch6 | 6 | 2e-4 | 444 | 1,363,968 | 1.8907 | 1.8181 | 24.94% |
| **`lr3e4`** | **4** | **3e-4** | **592** | **1,212,416** | **1.7835** | **1.7489** | **26.86%** |

The larger batches processed more tokens but took fewer optimizer steps and
were worse.  The higher LR is the clear winner: **-0.0425 test BPB and
+0.93 top-1 points** versus the 2e-4 reference at essentially the same token
budget.  The object gate remained closed (0.01993).

The current recommended preset is therefore `kaggle10m_lr3e4`.  The next
checks are a 600-step WSD decay at 3e-4 and a 4e-4 arm.

## Next-move arms

Run: <https://www.kaggle.com/code/shem42/z4-115m-next-moves>  
One kernel, same WikiText-2 bins, three sequential 600-second arms.

| Arm | Params | Steps | Tokens | Best val BPB | Test BPB | Test top-1 | Object gate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `kaggle10m` reference | 115,238,237 | 599 | 1,226,752 | **1.8254** | **1.7914** | **25.93%** | 0.01943 |
| `kaggle10m_obj` (gate init 0.05) | 115,238,237 | 576 | 1,179,648 | 1.8248 | 1.7989 | 25.82% | **0.05237** |
| `kaggle10m_route` (curriculum gate) | 115,238,237 | 597 | 1,222,656 | 1.8273 | **1.7910** | 25.87% | 0.01934 |
| `kaggle10m_mtp64` (rank 64) | 111,097,437 | 586 | 1,200,128 | 1.8311 | 1.8012 | 24.74% | 0.01918 |

Interpretation:

- **Forcing the object gate open did not help.**  The gate reached 0.0524, so
  the path was active, but test BPB regressed by 0.0075.  A closed object gate
  is not the only reason the object memory is weak at this horizon.
- **Route gating is a tie.**  Test BPB moved by 0.0004 (noise) and validation
  BPB was 0.0019 worse.  It is safe but not a win.
- **Rank-64 MTP did not pay.**  It saved 4.14M parameters and some head
  compute, but test BPB regressed by 0.0099 and top-1 fell 1.2 points.  The
  rank-128 MTP output stays.
- The reference `kaggle10m` remains the best short-run profile among these.

The next bottleneck is not these four knobs.  It is the **token budget**: the
reference sees 1.23M tokens in 600 seconds.  The next round should test batch
6/8 with the memory savings (8→6→4 fallback), and a 3e-4 LR at batch 4, before
adding more architecture.

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

The token and route feedback paths opened.  The token gate is the only strongly
open gate and it is **inhibitory** (-0.0375); the route gates are also
negative.  The memory gate is weakly positive.  This shows the bus is active
and trainable, not that it helped the loss.  The object gate is still
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
- The stock `115m` same-hardware baseline is above.  It OOM'd at batch 4 and
  ran at batch 2, so the comparison is the practical profile difference, not a
  clean single-variable ablation.

## First attempt

The first Kaggle attempt was stopped after its data-prep fallback converted a
2.05 GB FineWeb-Edu parquet to 3.47 GB of text and then tried to build bins
from the whole file.  The public Kaggle range URL returned 404 inside the
kernel.  The fixed notebook uses WikiText-2 for the fast gate; FineWeb should
be pre-sliced or pre-tokenized before a repeat run.
