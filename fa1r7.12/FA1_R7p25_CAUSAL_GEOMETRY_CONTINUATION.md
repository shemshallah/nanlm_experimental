# FA1 R7.25 — single-lineage geometric Diamond continuation

**Status: source implementation and CPU compatibility audit complete; model UNPROMOTED.**
This is the same R7 model in `fa1r7.12` on GitHub `master`. No second
network, separate branch, reset optimizer, or acceptance decision is created.

## Recovered starting point

The genuine 384-update R7.24 candidate and original R7.13 full recovery root
were recovered from their existing archives and checked locally.

- Accepted R7 checkpoint: **step 14,416**, SHA-256
  `54e835b00642013d4f1bf486a8150eb1ff06c547565771bd21933e143d050f2d`.
- Source unpromoted R7.24 candidate: **384 geometry updates**, SHA-256
  `5d07b1ef66aa9f870d05c6fda257b1dcdf36ecd4623999dff4a347cf82e4ed65`.
- The source candidate predates spherical face inputs and tetra-conditioned
  specialist matrices: it has a 9-input face projection, not the canonical
  11-input projection. Its five optimizers (eight groups in the fifth) were
  genuine and preserved rather than discarded.

`fa1_r724_unified.load_geometric` now supports this exact historical layout:

1. Copy the old face projection's first nine weight columns and Adam moment
   columns byte-for-byte. Initialize the two new spherical columns and moments
   to zero, leaving the old function identical.
2. Initialize the new 256 sparse apex-specialist operators with their output
   gate at zero, leaving the old function identical.
3. Map original Adam slots by **parameter name**, not by changed positional
   registration order; retain the other four optimizers exactly.
4. Reject unknown checkpoint layouts and refuse automatic promotion.

The old and migrated geometric Diamond produced **three exactly identical
outputs (maximum delta 0.0)** on the same synthetic causal inputs. A separate
roundtrip checked all old geometric Adam tensors and frozen R7 tensors.

## Corrected causal evaluation

The previous R7.24 code closed only three geometric gates when constructing its
baseline. Once the 256 sparse meta specialists learn, this would leave the
fourth, `meta_gate`, active in the purported geometry-off control. R7.25 closes
**all four** gates and reports independent edge, face, tetra, and apex-specialist
ablations. The historical R7.24 test reports are retained but are **not**
retroactively relabeled as four-gate experiments.

R7.25 also adds deterministic **different-book donor** controls. A complete
target-free Diamond context (32 past words, 128 past bytes, 8 sentence anchors,
8 vertex addresses, 92 near/far features, and 12 masked modal tokens) comes
from a *single other book*. The ground-truth next word, inherited hidden state,
inherited near/far predictor, and original book identity remain unchanged.
The test fails closed if fewer than two books are available. Book-level
bootstrap intervals remain required; context donor swapping is not a semantic
understanding proof.

`fa1_r724_train.py` can now resume the saved candidate using
`--resume-geometric`, with all existing geometry Adam histories, while keeping
the 14,416-step accepted ancestor unchanged. It refuses to overwrite its
input checkpoint and requires a book-labeled NaNROUTER corpus. Resume selection
must improve over both the geometry-off baseline and the *starting* candidate.

## CPU evidence and limitations

Verified locally with PyTorch 2.5.1 CPU:

- Sparse geometry tests: 70 distinct local tetrahedra, signed S+/S- parity,
  `d²=0`, nonzero trainable tetra residues, no pair/point bypass, differentiable
  256-sector meta specialization, four-gate parity: **PASS**.
- Different-book donor tests (including imbalanced books, deterministic
  sampling, unchanged target and inherited state, single-book rejection):
  **PASS**.
- 384-step candidate read, exact inherited weights/moments and migration
  zero-delta prediction parity: **PASS**.

### Read-only historical corpus smoke

The following samples come from the original *historical* R7 corpus, not a
new, book-disjoint NaNROUTER holdout. Values are next-vocabulary-word NLL in
**nats/word, not bits per byte**. Positive geometry gain is better.

| Historical target positions | Full geometry NLL | All-four-gates-closed NLL | Geometry gain |
|---:|---:|---:|---:|
| 8 | 4.711058 | 4.702622 | **−0.008435** |
| 128 | 4.974302 | 4.971664 | **−0.002637** |
| 1,024 | 5.137118 | 5.136243 | **−0.000874** |

At 1,024 positions, independently zeroing the edge, face and tetra gates
lowered NLL by 0.000182, 0.000565 and 0.000169, respectively. The new
meta-specialist gate is zero on this recovered candidate (it did not exist
at training time), so its ablation is identically zero. These positions are
neither sealed nor book-clustered, so they are a regression *warning*, not a
definitive generalization estimate.

**No candidate promotion, no claim of frontier performance, and no GPU
escalation.** The next scientifically meaningful training run requires a
verified whole-book-disjoint NaNROUTER corpus and a parameter-matched
non-geometric control. Train on TRAIN only; choose among new candidates on DEV;
test the selected candidate once on truly unseen books. Require positive
book-level confidence bounds, correct-history superiority, S- and tetra
ablations, raw-byte BPB, and optimizer continuity.

## Reproduction

```bash
export PYTHONPATH=fa1r7.12/source:/path/to/accepted_r7_root/source
python fa1r7.12/source/test_fa1_r724_geometry.py
python fa1r7.12/source/test_fa1_r725_controls.py

# Resume only with a verified, book-labeled NaNROUTER corpus.
# Output must be a NEW candidate path, not the protected source file.
python fa1r7.12/source/fa1_r724_train.py \
  --root /path/to/accepted_r7_root \
  --resume-geometric /path/to/FA1_R7p24_GEOMETRIC_UNPROMOTED.pt \
  --corpus-dir /path/to/book_disjoint_nanrouter/processed \
  --checkpoint /path/to/FA1_R7p25_UNPROMOTED.pt \
  --output /path/to/R725_cpu.json \
  --train 384 --dev 160 --steps 128 --minibatch 24 \
  --require-book-ci --threads 2

# Independent positions; now includes all four geometry gates and
# foreign-book donor comparison. Not a sealed-test promotion command.
python fa1r7.12/source/fa1_r724_eval.py \
  --root /path/to/accepted_r7_root \
  --candidate /path/to/FA1_R7p25_UNPROMOTED.pt \
  --corpus-dir /path/to/book_disjoint_nanrouter/processed \
  --split dev --targets 256 --seed 19025
```
