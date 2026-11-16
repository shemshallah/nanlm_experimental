# FA1 R7.51 — bidirectional exact-pair evidence and calibrated control

Author: Justin Howard-Stanley · Independent Researcher · Santa Cruz, California, USA.

**Status: implementation candidate, NOT evaluated or promoted.** The R7.49
checkpoint is the immutable parent. R7.50 was rejected for lack of positive
whole-book DEV attribution. This document describes new source/protocol; it
does **not** report an R7.51 predictive gain.

## Hypothesis

R7.50 found positive association between exact pair membership and gold words,
but its negative inhibitory coefficient improved TRAIN loss. These are
consistent if an inherited copy score is already overconfident. R7.51
separates two identifiable mechanisms:

1. **Global score normalization** applies a bounded learned temperature to
   pre-existing pair logits. It does not access address-specific new values.
2. **Bidirectional evidence** reads nonnegative support and opposition from
   the exact immutable C(256,2) candidate address; their signed difference is
   gated by a pre-target apex context, prior-word repetition, coverage and
   triangle-attention concentration. A global pair offset is retained as a
   separately ablatable non-address control.

The positive and negative banks are 32,640 scalars each; no new LM/decoder.
The inherited Diamond, signed S+/S− rails, and all 29 optimizer groups remain.

## Migration and training protocol

- Source: **R7.49** SHA-256
  `b8dabe54e2417a75defd01c776606367a3f3b0f18f479b032f8390b6fc87ddb5`.
- At gain=0 and temperature=0, require bitwise equality of R7.49 outputs,
  29 original optimizer groups and all saved Adam moments unchanged.
- Append one new AdamW parameter group, reaching **30 groups** in one
  continuing optimizer. Never reset or average inherited moments.
- Stage 1: TRAIN-only score-temperature calibration (default 32 updates).
- Stage 2: TRAIN-only signed pair-address refinement (default 96 updates).
  The base model remains frozen. The same next-word NLL supervises both stages;
  no gold label is used by confidence, routing or evidence features.
- Require SHA-pinned NaNROUTER **270 TRAIN / 15 DEV** identities and
  at least 150 TRAIN books in the sampled pool; balanced minibatches select
  distinct TRAIN books where possible. The 15 TEST books are not read.
- Make an immutable unpromoted R7.51 checkpoint and comparison report only
  after the program completes. Preserve the full optimizer and source SHA.
- The *implemented* CI only covers code and unit-level causality/parity
  regressions. A real checkpoint-backed run still needs corpus preparation,
  separate whole-book DEV evaluation, long-context and sealed TEST gates.

## Counterfactual evidence required for promotion

All values are on identical held-out DEV word positions and computed with
whole-book bootstrap confidence intervals:

| Comparison | What it isolates |
| --- | --- |
| R7.49 vs complete R7.51 | Total next-word NLL gain |
| R7.51 calibration-only vs full | Incremental signed geometric evidence |
| Correct vs substituted active addresses | Causal address specificity |
| Constant bank vs learned 32,640-address bank | Address-value necessity beyond a global exact-hit bias |
| Permuted learned bank vs correct bank | Identity of stored pair-address values |
| Foreign history vs correct history | Causal history sensitivity |
| Fixed vs learned confidence | Whether context calibration contributes |

**Strict DEV qualification:** positive mean and positive lower 95% whole-book
bootstrap bound for total word gain, improvement over calibration-only,
correct-address specificity, constant-bank necessity, row-permutation necessity
and correct-history advantage. A positive raw NLL alone is insufficient.

DEV is selection-only. Independently held-out TEST remains sealed until those
gates pass and the candidate is frozen, then long-context and full-system
regressions must also pass before any promotion. Full normalized next-word
top-1/top-5, raw-text bits per byte and +2/+3 metrics have not been measured
by this code and must not be invented.

## Implementation

- `source/fa1_r751_bidirectional_evidence.py`
- `source/test_fa1_r751_bidirectional_evidence.py`
- `source/fa1_r751_cpu_pilot.py`
- `source/fa1_r727_checkpoint.py` (R7.51 full-checkpoint reload migration)
- `.github/workflows/fa1-r751-bidirectional-evidence.yml`

This experiment does not modify FA2, M3, Terminal-RSI, accepted FA1 R7.13,
or the protected R7.49/R7.50 checkpoint archives.
