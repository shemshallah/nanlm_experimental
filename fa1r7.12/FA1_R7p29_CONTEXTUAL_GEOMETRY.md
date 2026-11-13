# FA1 R7.29 — Context-addressed sparse geometric memory

**Development policy:** `master` only, one R7 model, one optimizer, inherited
learned tensors and optimizer histories intact. No independent branch or model
replacement. No automatic promotion. The R7.28 successor remains a rejected
research checkpoint; the canonical accepted step-14,416 ancestor is protected.

## Why R7.28 did not generalize

R7.28 trained 34K+ signed face-code rows but worsened full word likelihood on
six external TEST books despite improving auxiliary first-byte top-5. The
R7.24 geometry produced mostly static address features and a small residual
that the main word decoder could not reliably trust. Sparse writes and
triangular curvature are not in themselves evidence of useful predictions.

## R7.29 architecture

The existing context-conditioned Diamond already computes a 32D state from
past words, bytes, sentence starts and inherited near/far document memory.
R7.29 uses that state to **query**, rather than only activate, the 56 signed
triangles connecting the active eight points:

1. Address all 56 real C(256,3) triangles. Retain their original 3D trainable
   codes and both R7.24 persistent signed face-code embeddings, S+ and S−.
2. Score each triangle against the target-free current context. Signed face
   values modify attention keys and retrieved values.
3. Retrieve the best **eight** triangles by sparse attention, without
   allocating a dense 2.76M-triangle attention matrix.
4. Return a *non-cancelling* weighted memory value to those triangles, then
   use the **existing** apex → triangle → pair → point feedback and word tree.
5. Train a zero-initialized scalar `context_read_gain` as a 14th parameter
   group within the **same** unified optimizer. The original 13 groups retain
   their values, groups, moments and parameter identities. A zero gain gives
   bit-identical R7.28 predictions.

### Critical cancellation repair

The initial development prototype sent `alpha_i * (retrieved - triangle_i)`
into each triangle while `retrieved = sum_i alpha_i * triangle_i`. Its
attention-weighted correction summed to zero, so the apex saw almost no
useful retrieval. The corrected operator sends the **retrieved signed value**
itself, distributed to eight addresses. This is not a cosmetic gate change;
it fixes the path that was cancelling its own information.

## Migration audit

The restored R7.28 full checkpoint passes an exact function-preserving
migration. All 231 inherited parameter objects/tensors remain unchanged.
The one optimizer changes from 13 to 14 groups, appending only the new scalar.
The new memory read has a finite, nonzero gradient from next-word likelihood
even at its zero-initialized state. `fa1_r729_parity.py` reproduces this on
actual checkpoint weights and NaNROUTER books.

## New corpus and evaluation contract

- 300 **previously unused** books from the fourth NaNROUTER short-story
  archive, SHA-disjoint from all 240 R7.28 training and evaluation books.
- 270 new TRAIN books, 15 new DEV books, 15 sealed TEST books. Source hashes
  and split manifests are preserved.
- Training still supervises complete next-word NLL, the next-word first byte,
  correct-versus-foreign-book history, geometry-off, apex-off, pair-off, and
  now **context-read-off** ablations with unchanged targets.
- The contextual read must beat its own read-disabled control *and* pass
  positive book-cluster confidence bounds before an R7.29 promotion claim.
- Same caveat as R7.28: this is a vocabulary-word model with an auxiliary
  first-byte head. The full normalized mixture top-5 and genuine raw-byte BPB
  cannot be reported without an actual autoregressive byte decoder.

This architecture is a research-scale step, **not** a claim of frontier-level
language-model performance. Model capacity, data diversity, genuine
byte/BPE/word/sentence compositional decoding and independent transfer are
still necessary. Only a verified held-out result can justify GPU scaling.

## CPU reproduction

```bash
python fa1r7.12/source/fa1_r729_parity.py \
  --root R713_ANCESTOR --checkpoint R728_UNPROMOTED.pt \
  --corpus-dir NEW_NANROUTER_300_SPLIT --output R729_PARITY.json

python fa1r7.12/source/fa1_r728_geometric_memory.py --contextual \
  --root R713_ANCESTOR --checkpoint R728_UNPROMOTED.pt \
  --corpus-dir NEW_NANROUTER_300_SPLIT \
  --train 1024 --dev 256 --steps 80 --minibatch 16 --batch 8 \
  --eval-every 16 --ablate-every 4 \
  --output R729_DEV.json --candidate-checkpoint R729_UNPROMOTED.pt
```

GitHub Actions executes public-source unit and regression tests on `master`.
The real-weight results are produced locally because the Library checkpoint
has not been installed as a verified GitHub Release asset for runners.

## Result: corrected R7.29 CPU pilot, unpromoted

The first sparse-read prototype accidentally used a weighted self-subtraction
that nearly cancelled its own message. Its real-weight next-word gate gradient
was `-0.000748`. After replacing that with direct signed-value delivery, the
same zero-gate migration remained **bit-for-bit identical** and its gradient
was `-0.018375` (about 24.6 times larger). Both source-only geometry tests
passed; this is gradient reachability, not generalization.

The corrected version continued from the complete, unpromoted R7.28
checkpoint for 80 CPU updates on the **new 270-book TRAIN** split, choosing
step 80 by DEV full word loss. The unpromoted candidate SHA-256 is
`25b97bfb482211affd2a04306d8893b381ed7b263316c983caa7365995c2fbcd`.

| Fixed 512-position new DEV comparison | R7.28 inherited | R7.29 corrected |
| --- | ---: | ---: |
| Full-system word NLL (lower) | 5.151152 | **5.150056** |
| First-byte top-1 (higher) | **11.72%** | 10.55% |
| First-byte top-5 (higher) | **43.55%** | 42.58% |
| Geometric path ablation gain (nats/word) | +0.001078 | +0.000485 |
| Correct-book context advantage (nats/word) | −0.000641 | −0.000825 |

The candidate's paired mean full-NLL gain was **+0.001096 nats/word**, but
the four-book bootstrap 95% interval was **[-0.001034, +0.003401]**: not a
statistically reliable improvement. More decisively, disabling only the NEW
contextual read **improved** word loss by `0.00002942 nats/word`; the read-on
minus read-off benefit had a strictly negative confidence interval
**[-0.00003866, -0.00002057]**. The R7.29 readout is therefore **rejected**.
The 15 fresh TEST books were **not consulted**.

R7.29 updated 62,859 S+ and 62,857 S− exact-address codes and 83,894
pre-existing R7.23 triangle-code rows. This confirms optimizer writes but
does not establish causal predictive storage. The full 263 MB trained state
reloads successfully, with one optimizer containing 14 groups. Audit proved
all **105 non-Diamond weight tensors** and **95 pre-Diamond Adam-state entries**
bitwise identical to R7.28; the legacy accumulator also matched bitwise.

**Architecture lesson:** Higher triangle-attention gradient did not convert
stored addresses into useful generalizable memory. The next design must
separate (1) write-target learning, (2) context-to-address retrieval quality,
(3) readout trust and (4) end-to-end word/byte decoder contribution. A sparse,
factorized key-value representation with a self-supervised past/future
retrieval task and causal per-book confidence gating is preferable to further
blindly increasing a global residual gain. This remains one R7 architecture,
not a request for a second independent model.
