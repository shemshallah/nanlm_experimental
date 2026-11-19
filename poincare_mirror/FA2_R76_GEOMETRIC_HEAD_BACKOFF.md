# FA2 R76 — Confidence-weighted geometric head backoff

**Status: rejected at independent TRAIN-book confirmation; R69 stays active.**

## Architecture

`CausalBookSuccessor` now accepts an opt-in `head_backoff_weight`, defaulting
to 1.0 and reproducing inherited R69 output exactly. When some queried edges
have exact pair-byte evidence and others only shared-head evidence, the new
coefficient changes the relative probability mass assigned to the two types
before the inherited R69 trainable successor rail sees the histogram. A
coefficient less than 1 downweights shared-head matches; 1 is the original
equal-edge aggregation. Coverage, hit counts, source maturities and all
history book boundaries are unchanged. The mechanism is causal by construction.

This is *retrieval-parameter calibration*, not AdamW/neural gradient training.
It is deliberately kept separate from R74/R75 additive neural readouts.

## Auditable CPU calibration and testing

- Accepted parent: FA2 R69 step 4465, checkpoint SHA-256
  `b5e366bf2e81f8376660477aaba0d57ddcb7b31836e8203e7b959ff65f17c498`.
- SHA-256 verified whole-book NaNROUTER TRAIN corpus; DEV/TEST unopened.
- Eight books/1,024 scored positions for fitting; four different books/512
  positions for confirmation; 256-prefix warmup per book, 64-token model
  context, and five coefficients tested: 0.125, 0.25, 0.5, 1, 2.
- Fitted score: `CE1 + 0.35*CE2 + 0.35*CE3`; selected 0.125 from TRAIN.
- No model weights or optimizer tensors were altered. All 198 AdamW states
  remain available in the original R69 checkpoint.

| Confirmed on 4 separate TRAIN books | Parent R69 | Selected R76 | Direction |
|---|---:|---:|---|
| Next-byte CE (nats/byte) | 4.14298302 | 4.14356427 | worse |
| +2-byte CE | 4.65619648 | 4.65626733 | worse |
| +3-byte CE | 4.65261444 | 4.65292840 | worse |
| Next-byte BPB | 5.97706106 | 5.97789962 | worse |
| Next top-1 / top-5 | 12.50% / 30.66% | 12.50% / 30.66% | tied |
| +2 top-1 / top-5 | 15.82% / 24.61% | 15.82% / 24.61% | tied |
| +3 top-1 / top-5 | 17.58% / 22.85% | 17.58% / 22.85% | tied |

Although a number of fit books improved and R76 materially changed the
retrieval distributions in 19–84 sampled positions per book, the negative
independent-book result fails the promotion gate. The DEV comparison script
refuses to open DEV unless the TRAIN confirmation CE1 gain is positive.

Code: `scripts/calibrate_r76_head_weight.py`,
`scripts/compare_r76_dev.py`, `tests/test_r76_head_backoff.py`.
Data: `results/FA2_R76_head_weight_TRAIN_confirmation.json`.

## Follow-up hypothesis

R74–R76 attempted global corrections; all three failed to generalize. The
next experiment should learn per-address reliability from *past observed
successor outcomes*, then apply the reliability only to that same address.
This creates causal local specialization without conflating independent books.
It must compare unchanged R69, exact-address, wrong-address, wrong-successor,
and unrelated-book controls on the same held-out positions.
