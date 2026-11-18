# M3 R31 — class-calibrated, checkpoint-preserving CPU continuation

R29 remains the **unified accepted default**. The R30 readout was not promoted:
its measured DEV BPB improved but nonspace BPB and top-1 regressed, and its
matched unrelated history beat the candidate on the GATE books. R31 probes
whether that failure is predominantly class calibration rather than history.

## Architecture

R31 reuses the exact R29 causal state and trained R29 lexical readout. It
reconstructs the three trained R30 feature gains and splits their logit
correction into disjoint nonspace and space channels, adding a separate space
intercept. Its three newborn trainable gains are **zero-initialized**, so the
untrained R31 prediction is **bitwise identical** to accepted R29; the exact
trained R30 remains accessible as an explicit mode. Neither the causal state
nor the original predictor is forked. Space/nonspace labels are used only for
the training loss, never as inputs to inference before a target is observed.

## Recovery and split

Parent: SHA-256 `a757433f6b72b4b444763de2f3158eea960a530c94fe613af58de247647b3421`
(R30 full checkpoint from Actions run `44218221694`). The 25 inherited
coefficients, Adam first/second moments, per-group step counters, and 256-byte
unigram prior are validated and copied without modification. Three new Adam
groups are appended; the resulting NPZ contains 28 parameter/moment/age groups.

NaNROUTER Release `corpora-short-stories-0004`, canonical seeded book order:

| Split | Book indices | Use |
| --- | --- | --- |
| FIT | 412–427 | Train exactly three new gains |
| GATE | 428–431 | Independent screening, no parameter updates |
| DEV | 432–443 | Independent comparison and whole-book bootstrap |
| TEST | 444–467 | **Sealed; never loaded** |

Evaluation scores each target byte before observing it, with six arms: R29,
trained R30, R31, wrong successor, matched-prefix unrelated history, and
full-unrelated history. Report BPB, bits/character, nonspace BPB, byte top-1,
byte top-5, per-book gains, descriptive whole-book bootstrap and runtime.

R31 may not replace R29 unless it beats R29, R30, wrong successor and matched
unrelated history on both GATE and DEV; avoids nonspace and top-1/top-5
regressions; improves at least 3/4 GATE and 9/12 DEV books; and has a positive
DEV bootstrap lower bound. Passing these is a **candidate** stage only:
independently test on sealed TEST before any promotion. In particular, this
experiment does not by itself establish geometry-based generalization.

Run with:

```bash
python -m m3_r31.train_dev \
  --corpus-root /tmp/m3_r31_corpus \
  --parent-checkpoint /tmp/m3_r31_parent/R30_full_adam_checkpoint.npz \
  --output m3_r31/results/R31_DEV.json \
  --checkpoint m3_r31/results/R31_full_adam_checkpoint.npz
```
