# M3 R22 — independent signed-memory feedback and matched n-gram controls

M3 R21's default online predictor remains unchanged (R20) and was independently tested on 24 books: 73,728 scored bytes, BPB 3.837562 (R19) → 3.505226 (default R20), 24/24 book improvements. [Verified run #43977992731](https://github.com/shemshallah/nanlm_experimental/actions/runs/43977992731).

R22 is a new **candidate inside M3**, not an FA1, FA2, NaNLMind, or Terminal-RSI modification.

## Why another upgrade?

The R20 pair and triangle scales are both 0.5. Their log likelihood ratios telescope:

    .5 log(Ppair/Pbase) + .5 log(Ptriangle/Ppair)
    = .5 log(Ptriangle/Pbase).

This is mathematically exact. Ppair remains important for smoothing Ptriangle, but its independently weighted residual disappears. R22 adds three separate, causal, count-confidence-dependent feedback signals on top of the frozen R20 output:

1. Pair residual from Ppair/Pbase with pair-support gate n/(n+16).
2. Triangle residual from Ptriangle/Ppair with triangle-support gate m/(m+8).
3. Signed directed flux J(a,b)=(N(a,b)-N(b,a))/(N(a,b)+N(b,a)+1).

Every R22 coefficient can equal zero. A triple-zero configuration recovers the exact immutable R21 default. No forward-looking memory is used.

## Fair controls

The benchmark independently implements **ordinary directed bigram and ordered trigram counts**, with precisely the same 8,192-transition retention, 8/4 pseudocounts, causal update order, byte prior and inputs. Mathematical and byte-stream parity must be exact within 2e-10. The conventional count-mixture is *identical by isomorphism* to R20: there is no claimed geometric advantage over this control. The standalone conventional trigram likelihood is also reported.

The new signed flux likewise derives from ordinary directed counts. Demonstrating a benefit from its explicit S+/S− formulation is not evidence that the mathematical address permutation alone creates extra information. A capacity-matched neural model is not measured here without its actual weights and comparable evaluation interface.

## Frozen split

- Earlier R17–R18 TEST, R19 reserved TEST and R20 DEV remain excluded from candidate fitting.
- Independent R21 TEST: indices 132–155, SHA-256 fingerprints locked; never used for R22 fitting, coefficient selection or DEV.
- **R22 FIT**: indices 156–171 (16 new books).
- **R22 GATE**: indices 172–175 (4 new books).
- **R22 DEV**: indices 176–187 (12 new books).

Fit restricts a 27-option grid to six candidates and zero. Independent TRAIN-only gate chooses among those, subject to non-space non-regression. DEV then evaluates the fixed winner, with wrong-flux, wrong-pair and no-flux controls; whole-book bootstrap and all next-byte top-1/top-5/BPB/BPC/non-space metrics.

R22's candidate cannot replace R21 default on the basis of source tests alone; a positive fresh DEV and another untouched independent TEST are mandatory. The original R12 trained checkpoint is still not available; these are standalone M3, frozen oscillator/memory results.
