# M3 R14 — Causal Wave Readout and Held-Out Book Pilot

M3 remains independent of FA1, FA2, Terminal-RSI and NaNLMind's VM core.

This round makes a real wave **predictive** by reading its directed displacement,
momentum and reciprocal-head signals **before** observing each target byte.
The candidate log-probabilities are a strictly normalized residual correction
of the exact-memory distribution:

```
log P_candidate(b) = log P_memory(b) + gain * wave_score(b) - log Z
```

One signed scalar `gain` is selected from a prespecified grid using **TRAIN
books only**; it is frozen for book-disjoint DEV. If `gain=0`, predictions
equal the memory-only parent to floating point precision. No R12 inherited
neural/ridge checkpoint or optimizer was available or altered.

## Arms and controls

| Arm | 128+2048 memory | Oscillation | Spatial propagation | Head-pair spring |
|---|---|---|---|---|
| memory_only | yes | no | no | no |
| uncoupled | yes | yes | no | no |
| local_wave | yes | yes | yes | no |
| reciprocal_wave | yes | yes | yes | yes |

All three wave arms share damping, time-step, impulse and on-site oscillator
frequency; only graph movement and reciprocal springs differ.

`m3_r13/mirror_wave.py` now supports an optional on-site frequency. Its
default is zero, preserving R13 numerical diagnostics. R14 uses frequency
1 for each wave arm, giving a proper graph-disabled temporal oscillator.

The exact-memory baseline here is a **new explicit 128/2048 two-window
implementation**, not a recovered R12 code path. Results of this probe
cannot be compared directly with previously reported R12 BPB.

## Causality and isolation

1. Derive the prior from TRAIN books only.
2. Reset both temporal memory and wave field at each book boundary.
3. At each token position, score using only already-observed transitions.
4. Apply the observed target as a velocity impulse **after** scoring.
5. Fit gain against TRAIN targets only. Never use DEV targets to choose a
   gain, operator, time step, or any other configuration.
6. Report DEV BPB/BPC, non-space BPB, top-1/top-5, gain, throughput and
   the graph-vs-oscillator attribution separately. Do not promote on a
   small pilot alone; require independent larger book-split and context
   tests before declaring a winner.

UTF-8 prefixes are trimmed at full-character boundaries and scoring starts
at a character boundary, so BPC uses a strict count of Unicode code points
without replacement characters or invalid truncated sequences.

## Run

```bash
python3 -m unittest discover -s m3_r14 -p 'test_*.py' -v
python3 -m m3_r14.predictive_probe \
  --corpus-root /path/to/NaNROUTER/short-stories \
  --train-books 4 --dev-books 4 \
  --bytes-per-book 768 --warmup 256 \
  --output m3_r14/results/R14_pilot.json
```

The CI workflow checks the SHA-256 of the NaNROUTER short-story batch 0004
asset before extracting the public-domain text and evaluating. Corpus books
are disjoint between train and DEV; no separate TEST books are consumed.
