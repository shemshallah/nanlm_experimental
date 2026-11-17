# M3 R25 — trainable temporal feedback candidate

R24 is the accepted **unchanged** default. R25 adds one auxiliary, sparse,
causal successor memory to the same canonical online state and a three-scalar
trainable feedback projection. Its three feature channels are:

1. Two-byte context, with minimum three prior observations and strong smoothing;
2. Longest available 4/8/16/32-byte match among the last 256 transitions;
3. Longest available 4/8/16/32-byte match among the last 32,768 transitions.

All three channels combine **earlier observed successor counts** with R24's
current normalized byte distribution. The neural/wave/mirror/triangle/copy
parent is untouched. The zero-coefficient setting is **bitwise identical** to
R24, and all other coefficients are trained by a persistent 3-state Adam
optimizer. The wrong-successor (+37 cyclic shift) and full unrelated-book
retrieval controls have identical parent distributions.

The new R25 train/gate/development data are previously unused book indices
244–259 / 260–263 / 264–275, respectively, selected with the original R17
seed and first 4,096 bytes per book. Original TRAIN-only 0–7 derives the byte
prior. Previous R21 and R24 independent tests remain sealed from training.
**Future R25 independent TEST 276–299 is not even opened by the trainer.**

The GitHub workflow recovers the SHA-verified `NaNROUTER` batch 0004, runs
the predecessor regressions, then trains and scores a book-disjoint gate
and fresh DEV. It uploads a complete three-scalar checkpoint, Adam moments,
split hashes, whole-book bootstrap and top-1/top-5/BPB/BPC/nonspace metrics.
Even if DEV passes, R24 remains default until an independently frozen
readout is subjected to its own one-shot locked TEST.

Local command:

```bash
python -m unittest discover -s m3_r25 -p 'test_*.py' -v
python -m m3_r25.train_dev --corpus-root /path/to/verified/books \
  --output m3_r25/results/R25_DEV.json \
  --checkpoint m3_r25/results/R25_Adam_checkpoint.npz
```
