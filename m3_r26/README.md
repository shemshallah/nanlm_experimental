# M3 R26 — class-conditioned context-tree feedback

R25's real CPU training improved 36,864 DEV positions from 3.509647 to
3.505162 bits/byte, with top-1 37.009% → 37.801%, but its nonspace BPB
regressed 4.120100 → 4.121550. The four-book GATE also failed; R25 was
properly not promoted. This experiment directly addresses those failures.

R26 is **the same online M3**, not a new replacement backbone. It retains the
accepted frozen R24 predictor (signed mirrors, triangle feedback, trained
wave and 4/8/16-byte specialist), R25's causal near/long suffix memories,
and R25's complete trained 3-scalar Adam state. It adds:

- Hierarchical PPM-style byte context counts for orders 1, 2, 3, 4, 6, 8,
  interpolated against the R24 distribution, with exact strict causality;
- Four new residual channels: short syntactic context, medium phrase patterns,
  longer motifs, and a separate recent-versus-lifetime drift channel;
- Seven extra non-space-class correction channels so spaces cannot overwhelm
  the optimization objective;
- A non-space-weighted 14-scalar Adam continuation preserving all three R25
  inherited weights, moment vectors and optimizer step counts exactly.

The full R25 readout is recovered from the actual successful GitHub Actions
artifact from [R25 run #44125544449](https://github.com/shemshallah/nanlm_experimental/actions/runs/44125544449),
not recreated by retraining. At zero new coefficients R26 **exactly matches**
the recovered R25 distribution. At all-zero coefficients it matches R24.

TRAIN remains NaNROUTER books 244–259, GATE 260–263, independent DEV 264–275.
Earlier independent tests are never optimized, and potential new TEST 276–299
remains sealed. The workflow saves the full 14-weight checkpoint, all inherited
and new Adam moments and steps, book hashes, and parent/R25/candidate/control
BPB, BPC, nonspace, top-1/top-5 and whole-book confidence intervals.

This is an **experiment**, with no automatic promotion. It must improve both
overall and nonspace loss and beat R24, R25, wrong-successor and unrelated-book
controls on the frozen gate before it is a candidate for locked TEST.
