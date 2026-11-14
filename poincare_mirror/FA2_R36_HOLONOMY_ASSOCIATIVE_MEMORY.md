# FA2 R36 — Fourier phase-keyed triangle retrieval (experimental)

This round keeps the **existing FA2 model and all R35/R34 routes** intact.
There is no independent model, no new branch, and no checkpoint promotion.
Enable the additional route with `--holonomy-retrieval` when growing or
training a model that already enables `--triangle-curvature`; it starts with
zero residual contribution, so previous +1/+2/+3 logits are unchanged.

## Why R35's memory was weak

R35 pooled 56 active triangle states and stored their curvature-rotated
vectors in one causal running mean. That loses key-specific distinctions:
unrelated concept writes mix together. Moreover, Gauss--Bonnet curvature
defects can begin extremely close to zero when newly learned Poincare points
lie near the disk origin; pure curvature phases therefore initially have
little discriminatory power.

R36 adds a parallel **four-harmonic associative retrieval** and preserves
the R35 mean unchanged as an independent ablation route:

- An active triangle's angles and signed curvature defect are calculated as
  before. The key phase combines a rotation-invariant triangle angle
  difference `alpha-beta` with the oriented curvature phase.
- Trainable positive query/key strengths are combined with trigonometric
  modes `cos(w*phase)`, `sin(w*phase)` at `w = 1, 4, 16, 64`.
- At each position the state accumulates the causal sufficient statistics
  `sum(k*cos(w*phase)*v)` and `sum(k*sin(w*phase)*v)`. Each query retrieves
  a weighted sum using its own phase. The denominator is a positive
  unrotated key normalization, preventing sign instability.
- Queries read only **strictly earlier** writes; no write from the same or
  a later time position can enter the result. Each batch/document starts
  with an empty associative state.
- The added output has an independently initialized **zero residual gate**.
  An auxiliary next-byte head gives the retrieved state a gradient even
  while the residual gate is closed. The normal R35 and MoE pathways remain.

The state is `O(4 * geometric_rank)` per time step and does not allocate
millions of concept embeddings. Training currently materializes prefix
sufficient statistics for all positions, which costs
`O(context * 4 * geometric_rank)` activations and should be chunked/scanned
before very long 32K CPU training. The 256-point, 32,640-pair,
2,763,520-triangle and 174,792,640-tetrahedron address spaces are unchanged.

## Verified engineering results

`python -m unittest discover -s poincare_mirror/tests -p 'test_*.py' -q`

**67 tests passed, 2 skipped** in the local environment. Six new R36 tests
cover rotation-selective recall from otherwise identical histories,
strict causality/document isolation, exact inherited +1/+2/+3 logits,
identical old Adam state counts, direct gradient flow through
Fourier key/query/value and readout, and a complete 512D/36 mixed-head
forward/backward pass (with bounded depth and expert count).

### Matched synthetic grammar (960 additional updates per model)

Prior R35 weights and 57 Adam states were reused. Both candidates saw the
same training windows. Development: 240 positions, separate generated
motif documents. Reproduce with:

`python poincare_mirror/scripts/r36_holonomy_pilot.py 960`

| Metric | R35 continuing | R36 Fourier continuing |
| --- | ---: | ---: |
| next-byte CE (nats/byte) | 0.482136 | **0.477624** |
| bits/byte | 0.695575 | **0.689066** |
| next-byte top-1 | 91.67% | **92.50%** |
| next-byte top-5 | 96.67% | **97.08%** |

**Critical ablation:** disabling only R36's new retrieval at this endpoint
changes CE from 0.477624 to 0.477626, a difference of only 0.000002
nats/byte. Most of the model-level improvement cannot be attributed to
the direct retrieval pathway; auxiliary regularization may account for it.

### Delayed key-value recall (480 matched updates per model)

Each synthetic document defines fresh, random key-value associations.
Attention is restricted to eight positions, while geometric memory can
see the entire earlier document. Measurements are on 128 query targets
from 32 separately generated held-out documents. Reproduce with:

`python poincare_mirror/scripts/r36_delayed_recall.py 480`

| Metric | R35 continuing | R36 Fourier continuing |
| --- | ---: | ---: |
| query CE (nats/byte) | **1.359887** | 1.383529 |
| query bits/byte | **1.961903** | 1.996010 |
| query top-1 | 25.78% | **32.81%** |
| query top-5 | 100.00% | 100.00% |

Turning only the R36 retrieval path off changes query CE from 1.383529
to 1.383537 (an improvement of merely 0.000008 nats/byte when enabled).
This cannot establish actual learned associative retrieval. The lookup
test demonstrates the mechanism, but the language-model ablation does not.

## Promotion status

R36 is a **source-level, independently testable, unpromoted experiment**.
Neither the actual 512D R34 step-2,849 checkpoint nor its full 615-book
development splits have been restored in this environment. No comparison
to the historical R34 held-out language-model benchmark is claimed.

The next experiment must restore the exact optimizer-carrying R34/R35
lineage and distinguish three separate possible gains: (1) auxiliary
regularization, (2) R35 curvature mean, (3) R36 phase-keyed retrieval.
For long contexts, use independent-book and foreign-history controls,
top-1/top-5, +1/+2/+3 CE and bits/byte, and a nontrivial positive
component ablation before any checkpoint promotion.
