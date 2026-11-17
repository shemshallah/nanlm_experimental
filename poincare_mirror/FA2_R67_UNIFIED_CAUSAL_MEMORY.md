# FA2 R67 — unified causal memory, space/EOS-safe decision boundary

Justin Howard-Stanley · Independent Researcher · Santa Cruz, California, USA
17 November 2026 · `master` only · FA2 (not FA1 or Terminal-RSI)

## One checkpoint lineage

The protected R58 step-3,185 model, a 192D six-layer FA2 network, retains its
196 inherited AdamW state entries. R66 appends exactly one four-value joint
memory-trust weight to the **same** optimizer group, initialized at zero, and
performs 256 updates on 12 causal TRAIN books. The 3,072 distinct supervised
positions come after an observed 128-token book prefix. The original 196
states and every inherited neural parameter remain bitwise unchanged.

R67 keeps **all** R66 learned model weights and Adam moments exactly. Its
checkpoint configuration enables `pair_successor_space_rank_guard=True` in
the existing `PairSuccessorTrust` forward path. It adds zero parameters,
zero optimizer groups, zero specialists and no second inference model.

## Causal learned geometry

At query time, the same FA2 backbone produces hidden state, 257-way byte/EOS
logits and sparse C(256,2) geometric pair addresses. A persistent R62 index
provides a histogram of previously observed successor bytes, conditioned on
the query byte, pair addresses and optional shared heads. A memory record is
installed only after its successor has become the next observed input.
R55/R58 inherited rails remain active; the trained four-value joint trust
regulates only the conditional distribution over the 255 nonspace, non-EOS
byte outcomes. Neutral, wrong-address, wrong-byte and wrong-successor
controls use identical backbone logits and separate histories.

## R67 rank-safe coupling

The original joint rail preserves the total probability of literal space,
EOS and the nonspace group but can shift the *winning class* away from space.
R67 computes the largest convex blend of original and retrieved probability
distributions that retains a previously winning space/EOS class. If the
inherited winner is another byte, the learned memory is fully active. This
uses only current inherited/candidate logits, no labels or future history.

The same function is called by the unified model and by the explicit R63
read-only evaluation ablation. Tests cover probability normalization, EOS
protection, space winner preservation, changed nonspace rankings, true
model-forward integration and exact causal TRAIN-vs-DEV feature alignment.

## First measured 12-book DEV gate

128 observed warm-up bytes and 256 once-scored positions per book; 3,072
queries total. All confidence intervals use whole-book bootstrap with
Bonferroni correction across four controls. Independent TEST was not opened.

| Metric | Neutral | Exact R67 | Direction |
| --- | ---: | ---: | --- |
| Next CE (nats/byte) | 3.659344 | 3.442270 | lower by 0.217074 |
| Bits/byte | 5.279317 | 4.966145 | lower by 0.313172 |
| Nonspace CE | 3.966386 | 3.711743 | lower by 0.254644 |
| Top-1 next | 23.0143% | 23.8932% | higher by 0.879 points |
| Top-5 next | 45.1172% | 48.0143% | higher by 2.897 points |
| Space top-1 | 56.8960% | 56.8960% | exactly preserved |

Correct history outperformed neutral on next and nonspace CE for 10/12 DEV
books; the corrected confidence intervals for next CE improvement were
entirely positive against neutral, wrong-address, wrong-byte, and
wrong-successor controls. +2/+3 losses and accuracies were invariant to the
joint rail, since those frozen inherited heads are not modified by R67.

This is a genuine trained retrieval contribution, not evidence of frontier
language modeling. Broader deep-prefix and independent TEST validation are
required before declaring a general capability improvement.

## Reproducibility

Use the SHA-256-protected R58 checkpoint
`63cde9aa1875d9f5e3db78e92678b63057550584752cf20ffe8859c890895dc7`
and checked NaNROUTER-derived Gutenberg38 whole-book corpus:

```sh
python poincare_mirror/scripts/train_r66_causal_book.py \
  --checkpoint R58.pt --expected-parent-sha256 63cde9aa1875d9f5e3db78e92678b63057550584752cf20ffe8859c890895dc7 \
  --data-dir GUTENBERG38 --candidate R66.pt --report R66_train.json

python poincare_mirror/scripts/enable_r67_space_guard.py \
  --checkpoint R66.pt --expected-sha256 SHA256_OF_R66 --output R67.pt

python poincare_mirror/scripts/eval_r63_book_stream.py \
  --checkpoint R67.pt --expected-parent-sha256 SHA256_OF_R67 \
  --data-dir GUTENBERG38 --warmup-positions 128 \
  --max-book-positions 256 --dev-books 12 --output-report R67_dev.json
```

The checkpoint and optimizer are not replaced by a source-only CI run.
