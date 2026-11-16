# FA2 R50 — Geometric address integrity in the inherited reciprocal Diamond

Date: 2026-11-16 (UTC). Track: **FA2 Poincaré Mirror only**.

## Why this ablation exists

R47 showed that suppressing the outer reciprocal residual almost restores the
protected baseline, and R48's 32-update context-64 geometry-only continuation
did **not** pass the DEV promotion gate. R49 detected no statistically
established *reciprocal-specific* correct-history advantage on its deliberately
small next-byte target set. None of those experiments establishes that a
trained language model uses the actual C(256,2)/C(256,3) addresses.

R50 holds all checkpoint tensors fixed and tests an alternative explanation:
perhaps a reciprocal bridge is useful merely because it adds generic nonlinear
capacity, irrespective of which conceptual address it reads or returns to.

## Prespecified controls

All six modes share a single loaded checkpoint, the same sorted eight active
points per token, the same causal input sequence, the same trainable tensors,
and the same future prediction head:

| Mode | Alteration *at inference only* | Question |
|---|---|---|
| `exact` | None | Inherited reference |
| `pair_address_scrambled` | Fixed bijection of 32,640 stored pair-embedding lookup IDs, shared by S+/S− | Is the exact pair address-to-content association useful? |
| `triangle_address_scrambled` | Fixed bijection of 8,192 learned triangle-hash lookup IDs; exact Fourier component preserved | Does learned triangle-address memory help? |
| `incidence_scrambled` | Consistently relabel every 8/28/56/70 incidence row/column; token slots remain untouched | Does propagation require the *correct alignment* between point, pair, face and tetra slots? |
| `all_scrambled` | All three above simultaneously | Does the aligned combination matter? |
| `reciprocal_disabled` | Suppress the full reciprocal processing path | Inherited-transformer anchor |

The incidence shuffle satisfies `d2' @ d1' == 0` and `d3' @ d2' == 0`
**exactly**, as in the original chain complex. It is not a valid relabeling of
the *entire* model: the point/edge/face token arrays are intentionally left in
their original order, so their correspondence to the chain complex is broken.
That makes it a controlled misalignment test, not a test of topology's physical
existence, nor proof that triangles have linguistic meaning.

Pair addresses and triangle-hash addresses are shuffled independently with
seed `2026 + 50017`. Every instance of the existing geometry uses the same
permutation. No additional model, specialist or optimizer is created.

## Data, metrics and uncertainty

The probe accepts **only** the checksum-verified whole-book NaNROUTER DEV
split. The locked protocol scores 12 books, eight same-book 64-token windows
per book, or 6,144 positions per mode. The same causal windows are used for
next-byte, +2 and +3 expected-byte predictions. The 257-symbol vocabulary
includes literal bytes 0–255 and EOS=256. Report all following:

- Next/+2/+3 cross-entropy (nats per 257-symbol target), top-1 and top-5.
- Non-space next-byte NLL, top-1 and top-5; space-only as a diagnostic.
- Bits per byte-or-EOS target (`ce_1 / ln(2)`) and total bits divided by
  **literal bytes only** (excluding EOS in the denominator). These are
  distinct and must not be conflated with compressed-file BPC.
- Raw-byte denominator and measured model scoring throughput.
- Five exact-vs-perturbed comparisons using 2,000 *book*-cluster bootstrap
  replicates and simultaneous two-sided Bonferroni-adjusted confidence
  intervals; positive `vs_exact.mean_gain` means that the perturbed arm
  improves over exact, so **negative** means aligned addresses perform better.

The windows are sampled from independent books, but windows within a book
are not independent bootstrap units. No independent TEST exposure, training,
optimizer mutation, or automatic checkpoint promotion is permitted here.
Because each arm is evaluated under a deliberately invalid addressing
intervention, these scores are *mechanistic diagnostics*, not deployable
successor checkpoints.

## Required verification

1. Reversible inference-only pair and triangle pre-hooks, including exception
   cleanup and unchanged stored tensors.
2. Exact algebraic `d2@d1=0`, `d3@d2=0` under incidence shuffling.
3. Full replay of the original `exact` mode after **all** interventions, with
   byte-for-byte identical per-book metric floats.
4. Existing R47/R49 regression contracts and new R50 unit tests in GitHub CI.

## Outcomes

The measured DEV report is kept as
`results/FA2_R50_geometry_address_integrity_DEV_20261116.json`.

Actual protected-parent step: **2,993**. Verified 12 DEV books × 8 windows ×
64 targets = **6,144 next-byte prediction positions per mode**. The final
replay reproduced every per-book baseline metric *exactly* after restoration.

| Mode | Next-byte NLL (nats) ↓ | Bits per byte/EOS target ↓ | Next-byte top-1 ↑ | Top-5 ↑ | NLL gain *of scrambled over exact* (nats) |
|---|---:|---:|---:|---:|---:|
| Exact | 2.357199291 | 3.400719728 | 35.5143% | 70.4915% | — |
| Pair address scrambled | 2.357199172 | 3.400719556 | 35.5143% | 70.4915% | +0.000000119 |
| Triangle address scrambled | 2.357201288 | 3.400722609 | 35.5143% | 70.4915% | −0.000001997 |
| Incidence scrambled | 2.357164929 | 3.400670154 | 35.5143% | 70.4915% | +0.000034362 |
| All scrambled | 2.357166906 | 3.400673006 | 35.5143% | 70.4915% | +0.000032385 |
| Reciprocal disabled | **2.356905798** | **3.400296307** | 35.5143% | **70.5892%** | +0.000293493 |

The full measured report also contains +2/+3, non-space and space-only
predictions, raw-byte denominators, throughput and all 95% simultaneously
adjusted whole-book bootstrap intervals.

**Interpretation:** Scrambling learned addresses does *not* meaningfully
degrade this inherited checkpoint. The minute pair-ID contrast is approximately
1.2×10⁻⁷ nats/target in the *wrong* direction for useful aligned addressing.
Its bootstrap interval narrowly excludes zero, but it is too small to provide
evidence of a practically useful mechanism and must not be overstated. Both
incidence and all-address shuffles appear marginally *better* than the exact
path, although their corrected intervals include zero. The reciprocal-disabled
comparison also favors disabled on these sampled windows, but its adjusted
overall CE interval includes zero. No geometry promotion passes.

The current inherited `reciprocal.gate = −0.000367079745`, which implies a
global cold gate `tanh(8g) ≈ −0.0029366`. Learned simplex gates in participating
blocks are about −0.0123, −0.0142 and −0.0152. The tiny influence of address
shuffling is therefore consistent with suppressed feedback; it **does not**
establish that the representation could never become useful after training.

Next experiment: on TRAIN only, explicitly contrast predictions from correct
vs shuffled address alignment using the *same checkpoint* and preserve all
Adam moments. Evaluate its candidate with these exact R50 controls, then with
book-disjoint long-context checks. No independent TEST until the DEV gate has
been satisfied and the candidate has a complete recoverable checkpoint.
