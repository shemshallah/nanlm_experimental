# Canonical M3 online predictor — R29 directed default with certified R26 fallback

M3 now defaults to the combined R21/R24/R26/R29 architecture: signed mirror
pair/triangle successors, causal prefix copy, R26's near/lifetime context
tree, and R29's trained word-conditioned lexical successor feedback. **R26 is
the latest independently TEST-certified checkpoint.** R29 was made the default
at the user's explicit direction despite its unrelated-history control
outperforming it on GATE/DEV. Do not call this an R29 TEST certification.
This remains separate from FA1, FA2, NaNLMind and Terminal-RSI.

```python
from m3_default import M3DefaultPredictor, M3R21Predictor
default = M3DefaultPredictor(original_r16_train_prior)  # "r29", user-directed default
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

The exact copy gain and entire five-scalar Adam optimizer moment history
are frozen in `m3_r24/r23_frozen_candidate.json`; the original R23 fitted
checkpoint remains recoverable from GitHub Actions. The newer R26 checkpoint
preserves all 14 Adam group moments and steps.

Use `default.predict_proba("r26")` or `"r24"` for exact inherited predictions;
construct `M3DefaultPredictor(prior, mode="r26")` or `mode="r24"` to omit the
newer auxiliary memory when not needed.

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
