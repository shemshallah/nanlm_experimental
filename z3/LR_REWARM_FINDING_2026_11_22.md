# Why the 12h block gained so little: LR re-warm (2026-11-22)

The second 6-hour block added 51.6M tokens but only **−0.0275 PTB**
(1.7624 → 1.7349), versus −0.1304 for the first block. The log shows the
cause is the learning-rate schedule, not data exhaustion.

## What the schedule actually does

`--lr-horizon time --decay-frac 0.1` uses WSD over the **per-invocation**
wall clock:

- first 90% of the 6-hour budget (5h 24m): **constant peak LR**
- last 10% (36 minutes): cosine decay to `min_lr_frac` (2e-5)

Parsed from `z4-causal-gpu12h.log`:

| Step | Elapsed | LR |
|---:|---:|---:|
| 17,410 | 4 s | 6.2e-5 (warmup) |
| 18,000 | 1,010 s | **4.0e-4** |
| 20,000 | 4,434 s | **4.0e-4** |
| 22,000 | 7,858 s | **4.0e-4** |
| 24,000 | 11,287 s | **4.0e-4** |
| 26,000 | 14,718 s | **4.0e-4** |
| 28,000 | 18,156 s | **4.0e-4** |
| 30,000 | 21,592 s | **2.0e-5** (annealed) |

## Why that hurts across blocks

Each block ends with a short anneal, which produces a large one-off drop,
and the next block immediately re-warms to 4e-4 and destroys it.

Best valid bits/token from the 6h block:

| Step | Bits/token |
|---:|---:|
| 6,000 | 6.4145 |
| 8,000 | 6.3241 |
| 10,000 | 6.1886 |
| 12,000 | 6.0983 |
| 16,000 | 5.9628 |
| 17,408 | **5.7821** (final anneal) |

Then the 12h block restarts at 4e-4:

| Step | Bits/token | Note |
|---:|---:|---|
| 18,000 | 6.0531 | worse than the 17,408 start |
| 20,000 | 5.9628 | re-traced the 6h value from step 16,000 |
| 22,000 | 5.9176 | |
| 24,000 | 5.8724 | |
| 26,000 | 5.8272 | still worse than 17,408 |
| 30,000 | **5.6240** | only after the final anneal |

The block spent most of its 5.4 stable hours re-earning loss it had already
reached, then got the real gain in the last 36 minutes. The final PTB gain
(−0.0275) is consistent with that reading. The intermediate evals use only
4,096 valid tokens, so individual points are noisy, but the pattern is
unambiguous and matches the LR trace.

## Fix for the next block

Do **not** re-warm to 4e-4 for another 5.4 hours. Two workable schedules:

1. **Conservative continuation:** `--lr 5e-5 --decay-frac 0.3`.
   Starts near the previous anneal, decays for the last ~1.8h.
2. **Moderate continuation:** `--lr 1e-4 --decay-frac 0.5`.
   More room to move, with half the block in decay.

A 1-hour probe at `--lr 1e-4 --decay-frac 0.5` plus PTB would test this for
~4,000 GPU-seconds before committing another 21,600. The expected effect is
much larger than another 6-hour block at 4e-4.

## Interaction with precision

The same run used `--amp-dtype bf16` on a T4. The trainer's own policy is
fp16 for compute capability < 8.0. `z4-gpu-amp-probe` measures bf16 vs fp16
at identical batch/ctx so the next block can use the faster path if it is
numerically stable.
