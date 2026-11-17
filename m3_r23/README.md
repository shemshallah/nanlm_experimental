# M3 R23 — trained dynamical geometric apex and long-prefix memory

R22 independently proved that its signed-pair and triangle *counts* are exactly
isomorphic to conventional bigram/trigram counts and that the extra signed-flux
gate failed on held-out DEV. We retain R21 as the protected default.

R23 tests a deeper hypothesis **inside that same default predictor**: can a
genuine evolving 256-node geometric field predict novel bytes beyond count
lookup? The new `M3R23Predictor` subclasses `M3DefaultPredictor`, reusing its
single causal history, R16/R19 wave weights and R20 memory. `theta=0` recovers
all 256 R21 probabilities exactly, as tested at every position of a byte stream.

## Coupled architecture

* 256 head amplitudes and velocities occupy a periodic 16×16 geometric lattice.
* A second-order damped wave propagates through neighboring heads after each
  observed byte. Oriented triangle circulation in the inherited antisymmetric
  S− wave rail injects balanced apex feedback after the target is known.
* Four geometric feature fields (amplitude, velocity, neighborhood contrast,
  signed triangle-area-weighted velocity) reach the same 256-byte output.
* A sparse 8,192-transition memory retrieves earlier exact observed successors
  for 4-, 8- and 16-byte suffixes, with confidence-weighted backoff to R21.
* One five-parameter trainable residual readout combines these features with
  the frozen R21 log-probabilities. An Adam optimizer retains both moment arrays.
* No target is ever observed before being scored. Each book has a reset local
  dynamic state, while readout weights and optimizer state persist during FIT.

## Scientific controls

* `r20`: frozen R21 baseline (zero-readout exact parity).
* `apex_torus`: true lattice geometry, five readout weights trained on FIT.
* `apex_shuffled`: independently trained **same-sized** recurrent state on a
  fixed shuffled placement of byte labels, to test whether geometry adds value.
* `copy_only`: ablation with the trained geometric weights zeroed.
* `wave_only`: ablation with copy weight zeroed.
* `wrong_copy`: exact same prefix matches, deliberately corrupted successor
  values, testing whether retrieved content contributes.

R23 does not claim geometric superiority from a positive copy-only result.
Recurrent spatial superiority requires a positive new-book gain against both
the R21 baseline and the separately trained shuffled-geometry control.

## Book-disjoint protocol

NaNROUTER release `corpora-short-stories-0004`, SHA-256 verified. Exactly 220
seed-selected documents; the first 52 R17/R18 hash-locked, R21 TEST hashes
132–155 unchanged. No earlier TEST/DEV documents are reused for fitting.

| Partition | Seeded indices | Documents |
|---|---|---:|
| R23 FIT | 188–203 | 16 |
| R23 GATE | 204–207 | 4 |
| R23 fresh DEV | 208–219 | 12 |

Training uses 48-position Adam updates on FIT. Gate chooses among a genuinely
predictive torus, copy-only, and frozen R21 fallback; it never reads new DEV.
Only after gate selection does DEV compare BPB, BPC, nonspace BPB, top-1/5,
whole-book bootstrap, wrong-value and shuffled-topology controls. Full trained
state plus Adam moments is exported as a recoverable `.npz` artifact.

Even a positive DEV does not automatically overwrite R21. An untouched
independent TEST and stronger capacity-matched neural comparisons must precede
checkpoint promotion. The original R12 trained checkpoint was not recovered.
