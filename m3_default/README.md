# Canonical M3 online predictor — R32 directed default with exact fallbacks

M3 now defaults to the combined R21/R24/R26/R29/R30/R31/R32 architecture: signed mirror
pair/triangle successors, causal prefix copy, R26's near/lifetime context
tree, R29's trained word-conditioned lexical successor feedback, R30's trained
three-scalar correction, R31's class-selective calibration, and the R32
immutable TRAIN-only lexical anchor with four trained gains. **R26 is the
latest independently TEST-certified checkpoint until R32's locked independent
TEST completes.** Both R29 and R31 were made
defaults at the user's explicit direction despite unrelated-history controls
outperforming them on GATE/DEV. Do not call R31 a TEST certification.
This remains separate from FA1, FA2, NaNLMind and Terminal-RSI.

```python
from m3_default import M3DefaultPredictor, M3R21Predictor
default = M3DefaultPredictor(original_r16_train_prior)  # "r32", user-directed default
certified = M3DefaultPredictor(original_r16_train_prior, mode="r26")
legacy = M3R21Predictor(original_r16_train_prior)    # exact "r20"
for byte in document_prefix:
    default.observe(int(byte))
    legacy.observe(int(byte))
new_probs = default.predict_proba()
old_probs = default.predict_proba("r20")
```

Both classes expose `predict_proba`, `predict_logp`, `observe`, `reset`, and
the original R19/oscillator, triangle-only and wrong-address ablations.
Predictions are pure reads; targets enter causal memory only **after** being
scored. Reset clears document state without altering learned parameters.

Frozen inherited parameters:

- R16 original oscillator/spatial readout coefficients.
- R19 space/nonspace scales `0.25` and `1.0`.
- R20 directed pair/ordered triangle scales `0.5` and `0.5`.
- 256 heads, 32,640 signed pair addresses, 8,192-transition triangle history.
- R24 4/8/16-byte suffix memory, 8,192-transition causal horizon.
- R24 learned copy readout gain **`1.0962571833587516`**.
- R26 fourteen-scalar class-conditioned causal readout, frozen exactly from
  the independently tested recovery checkpoint.
- R29 eight-scalar lexical feedback readout, embedded exactly from the
  768-update R29 checkpoint; all 22 Adam groups remain recoverable.
- R30 three-scalar corrective readout, with 25 optimizer groups retained.
- R31 three-scalar class-selective calibration, with all 28 optimizer groups
  retained in its exact recoverable checkpoint.
- R32 four-scalar disjoint TRAIN-derived lexical anchor, with **32/32 Adam
  moments, variances and step ages** in `r32_accepted_checkpoint.npz`, the
  complete self-contained SHA-256-verified inference and training recovery.

The exact copy gain and entire five-scalar Adam optimizer moment history
are frozen in `m3_r24/r23_frozen_candidate.json`; the original R23 fitted
checkpoint remains recoverable from GitHub Actions. The newer R26 checkpoint
preserves all 14 Adam group moments and steps.

Use `default.predict_proba("r31")`, `"r30"`, `"r29"`, `"r26"` or `"r24"` for exact
inherited predictions;
construct `M3DefaultPredictor(prior, mode="r26")` or `mode="r24"` to omit the
newer auxiliary memory when not needed.

## R32 user-directed default and frozen evaluation

R32 adds an immutable background lexical memory fitted exclusively on 16
NaNROUTER TRAIN books, not on DEV, GATE or TEST. Its four gains are
`(0.5927959113137101, -1.5, -0.20722426183204995,
0.294518336989888)`. Unlike the R29/R31 case, both R32 development
gates passed: BPB dropped **3.303148 → 3.097359** on 12 disjoint DEV books,
top-1 rose **42.483% → 45.185%**, top-5 **71.806% → 75.319%**, and all
12 books improved. Correct TRAIN anchor beats matched unrelated history
(3.292148 BPB) and wrong successor values (3.524173 BPB).

The exact checkpoint and DEV JSON are now tracked in source as immutable
recovery assets. The independent 24-book TEST indices 516–539 and criteria
were committed in `m3_r33/r32_frozen_candidate.json` **before** any TEST
reading. The user authorized default promotion. This does not independently
certify R32's generalization until that locked TEST has run successfully.
R31 and all inherited modes remain callable without alterations.

- [R32 CPU TRAIN / GATE / DEV](https://github.com/shemshallah/nanlm_experimental/actions/runs/44233287739)
- [Prelocked R32 independent TEST](../m3_r33/r32_frozen_candidate.json)
- [R32 user-directed promotion and provenance](R32_DIRECTED_PROMOTION.md)

## R31 directed engineering promotion

The exact trained R31 coefficients are `1.2195544888726837`,
`0.03536605045213109`, and `-0.8597706306738135`. On 12 held-out
DEV books, its BPB improved from R29 **3.410224** to **3.394562** and its
nonspace BPB from **4.057942** to **3.898143**. However, matched unrelated
history scored **3.386495**, R31 top-1 dropped slightly, and only 7/12
books improved. Its GATE and DEV research criteria therefore **failed**;
TEST books 444–467 remain sealed. The user explicitly requested R31 as the
runtime default while this research continues.

- [R31 exact 28-group training checkpoint and metrics](https://github.com/shemshallah/nanlm_experimental/actions/runs/44227555060)
- [Explicit directed promotion and candid evidence ledger](R31_DIRECTED_PROMOTION.md)

## R29 user-directed default and scientific limit

Trained eight new weights on 16 fresh NaNROUTER books (49,152 scored bytes,
768 Adam updates), then compared on four GATE and 12 DEV books, without
loading the sealed 24-book R29 TEST (332–355). R29 outperformed R26 on
DEV: 3.470453 → 3.458221 BPB, nonspace 4.155572 → 4.139543 BPB,
top-1 39.625% → 39.945%, top-5 69.964% → 70.195%, 10/12 books improved.
But unrelated-history BPB was **3.453804**, better than R29's 3.458221.
Thus R29's prespecified GATE/DEV acceptance criteria **failed**, and no
independent TEST has been opened. The exact predictor is now usable as the
requested default, but certified R26 remains accessible without modification.

- [R29 training and GATE/DEV](https://github.com/shemshallah/nanlm_experimental/actions/runs/44184424516)
- [R29 directed promotion and checks](R29_DIRECTED_PROMOTION.md)

## R26 independent TEST and promotion

The R26 candidate was trained on NaNROUTER books 244–259, gated on 260–263,
and selected on DEV 264–275. The complete checkpoint, readout weights,
split indices and criteria were locked before untouched TEST 276–299.

| Metric | Frozen R24 | Promoted R26 |
| --- | ---: | ---: |
| Bits/byte ↓ | 3.294720 | **3.239214** |
| Nonspace bits/byte ↓ | 4.126202 | **4.055948** |
| Top-1 accuracy ↑ | 41.5484% | **43.8543%** |
| Top-5 accuracy ↑ | 71.8892% | **72.2811%** |

Of 24 new books (73,726 scored bytes), 23 improved; BPB gain +0.055506,
descriptive book-bootstrap 95% interval [+0.046668, +0.063913]. Wrong
successor and unrelated-book controls were worse (4.275695 and 3.872974
BPB). All prespecified acceptance conditions passed. No parameters were
selected after TEST.

- [R26 CPU TRAIN / GATE / DEV](https://github.com/shemshallah/nanlm_experimental/actions/runs/44157452299)
- [R26 locked independent TEST](https://github.com/shemshallah/nanlm_experimental/actions/runs/44167452153)
- [Frozen R26 criteria](../m3_r27/r26_frozen_candidate.json)
- [Acceptance and promotion ledger](R26_ACCEPTANCE.md)

## Earlier R24 independently verified TEST

FIT 188–203; GATE 204–207; DEV 208–219. The readout was frozen **before**
scoring 24 new books, indices 220–243 (73,726 scored bytes):

| Measure | R21 | Accepted R24 |
| --- | ---: | ---: |
| Bits/byte ↓ | 3.43649 | **3.34084** |
| Top-1 ↑ | 35.50% | **39.33%** |
| Top-5 ↑ | 70.47% | **71.51%** |
| Nonspace bits/byte ↓ | 4.04009 | **3.95073** |

All 24 TEST books improved; gain +0.09565093 bits/byte. Descriptive
whole-book bootstrap 95% interval +0.08268 to +0.10677 bits/byte.
The wrong-successor-value control worsened to 3.90343 bits/byte.

- [Independent R24 TEST passed](https://github.com/shemshallah/nanlm_experimental/actions/runs/44001403746)
- [R23 TRAIN/DEV and recoverable Adam history](https://github.com/shemshallah/nanlm_experimental/actions/runs/43992588664)

## Limits and preservation

R23's 256-head toroidal recurrent wave did **not** beat shuffled geometry.
It is not included in the accepted R24 readout. Signed pair/triangle counts
are address-isomorphic to ordinary bigram/trigram counts. This is a proven
causal-prefix retrieval improvement, not a demonstrated geometric-computation
advantage over matched conventional models. The complete R21 predictor
and its outputs remain available as the frozen legacy class.
