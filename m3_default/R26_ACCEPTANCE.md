# M3 R28 — promotion of frozen R26 as the canonical predictor

Decision: **PROMOTED**. R26's class-conditioned byte-context tree and
multiscale successor memory are active in `M3DefaultPredictor`; exact R24
and R21 fallbacks remain available in the SAME online M3 architecture.

## Frozen provenance

- Candidate source commit: `7870f549c9d35d708d3d3615bc5504408da8e348`.
- CPU training, four-book GATE and 12-book DEV run: `44157452299`.
- Complete 14-parameter model and Adam checkpoint SHA-256:
  `deb112d4d9b25cb57bd68f689660550a315032bea42c5e9376dbcdeb4a5c97a9`.
- DEV report SHA-256:
  `0c65d3b3a8adcb96d2b129076f74eaefc49ee69b61e629be168e5c36d7e20100`.
- One-shot sealed independent TEST run: `44167452153`, artifact `14365004998`.
- Locked weights, optimizer steps, split assignment and criteria:
  `m3_r27/r26_frozen_candidate.json` (committed BEFORE TEST was opened).

Three inherited R25 Adam groups reached step 1,536, and eleven R26 groups
reached step 768; the full checkpoint preserves all moment arrays and steps.
The deployable model embeds only the exact frozen 14 scalar coefficients,
requiring neither checkpoint fetching nor Adam to make predictions.

## Sealed TEST, NaNROUTER books 276–299

All arms score each byte before observing it and reset at book boundaries.
Frozen TRAIN prior, 1,024-byte warm-up, 24 unseen books, 73,726 positions.

| Arm | BPB ↓ | Nonspace BPB ↓ | Top-1 ↑ | Top-5 ↑ |
| --- | ---: | ---: | ---: | ---: |
| R24 accepted | 3.294720 | 4.126202 | 41.548% | 71.889% |
| R25 rejected | 3.289029 | 4.126999 | 42.347% | 72.038% |
| **R26 promoted** | **3.239214** | **4.055948** | **43.854%** | **72.281%** |
| Unrelated history | 3.872974 | 4.764628 | 38.516% | 69.690% |
| Wrong successor | 4.275695 | 3.889197 | 29.767% | 61.994% |

- 23/24 books improved against R24.
- R26 improved BPB by 0.055506 (1.684% relative).
- Descriptive whole-book bootstrap 95% gain interval:
  [+0.046668, +0.063913] BPB.
- All prespecified gates passed; TEST did not change the readout.

## Implementation invariants

- `M3DefaultPredictor(prior)` streams R21, R24 and R26 in one causal state;
  `predict_logp("r24")` and `predict_logp("r20")` remain accessible.
- `M3DefaultPredictor(prior, mode="r24")` omits R26's extra memory for
  historical R25/R26 migration and memory-constrained legacy inference.
- Pure readout, causal updates, document reset, normalized distributions,
  and exact bitwise parity with the frozen `M3R26Predictor` are CI-gated.
- This establishes independently measured byte-prediction improvement, **not
  yet geometric superiority** against matched conventional PPM or neural
  competitors: signed mirror addresses are isomorphic to sequence counts.
