# Canonical M3 online predictor — R26 accepted multiscale default

M3's accepted default is the R21 signed mirror-pair/ordered-triangle successor
backbone, R24's exact causal prefix-copy specialist, and R26's trained
near/lifetime successor memories and hierarchical context tree. R26 passed
its own prespecified 24-book independent TEST.
This remains separate from FA1, FA2, NaNLMind and Terminal-RSI.

```python
from m3_default import M3DefaultPredictor, M3R21Predictor
default = M3DefaultPredictor(original_r16_train_prior)  # "r26", independently TEST-certified
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

The exact copy gain and entire five-scalar Adam optimizer moment history
are frozen in `m3_r24/r23_frozen_candidate.json`; the original R23 fitted
checkpoint remains recoverable from GitHub Actions. The newer R26 checkpoint
preserves all 14 Adam group moments and steps.

Use `default.predict_proba("r24")` for the exact inherited R24 distribution;
use `M3DefaultPredictor(prior, mode="r24")` to construct an R24-only instance.

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
