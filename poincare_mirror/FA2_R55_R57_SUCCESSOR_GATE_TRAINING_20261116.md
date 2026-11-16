# FA2 R55–R57 — Exact geometric successor memory: genuine TRAIN-only gate learning

**2026-11-16 · Poincaré Mirror FA2 only · GitHub `master` · no promotion**

## Recovery and protocol

The previous R54 source had passed synthetic causality and optimizer-migration
tests but had not trained on its real historical checkpoint. This round
recovered and SHA-256 verified the exact R53/R54 parent:

`74478d2f77b3c2da7978458f3660dffe8aad788938ce4a7ab6eae7cff2de0817`

It also recovered the book-disjoint NaNROUTER/Gutenberg38 byte/EOS corpus.
The TRAIN and DEV files match their stored SHA-256 checksums, and the split
manifest is verified. There are 12 TRAIN, 12 DEV and 14 sealed TEST books.
**No TEST bytes were opened for these experiments.**

R55 attaches R54's four zero-initialized trust parameters to the existing
model and single AdamW optimizer group. The inherited 198 model-state tensors
and **all 194 inherited optimizer states remain bit-exact**. Only the newly
attached trust tensor receives gradient updates. The exact zero-trust state
preserves inherited next-byte predictions. Model architecture, tokenizer,
history-bearing optimizer state, +2/+3 heads, and book assignments stay fixed.

All three predeclared TRAIN objectives use 96 updates, 8 sampled 64-position
windows per TRAIN book, batch size 2, learning rate 0.02, and identical seeds.
The three values of the TRAIN-only non-space loss weight are 1× (R55),
2× (R57) and 4× (R56). These are **alternative, unpromoted diagnostic arms
from the same protected parent**, not independent replacement models and not
an optimizer lineage promoted over its ancestor.

DEV uses the same 8 sampled 64-position windows per book, or 6,144 prediction
targets per arm. The four matched inference conditions are: exact historical
addresses and observed successor bytes; zero-gate protected baseline; wrong
query-only addresses; and wrong successor values while preserving exact hit
support. Paired uncertainty resamples 12 whole DEV books 2,000 times, with
Bonferroni correction across the three contrasts within each training arm.
**Across-arm DEV selection is exploratory**: the three-arm search is not
covered by those within-arm intervals.

## Measured DEV scorecard

| Condition | Overall next CE ↓ | Non-space next CE ↓ | Bits/raw byte ↓ | Next top-1 ↑ | Next top-5 ↑ |
|---|---:|---:|---:|---:|---:|
| Protected parent | 2.357199 | 2.617379 | 3.400720 | 35.514% | 70.492% |
| R55: weight 1× | **2.351676** | 2.617146 | **3.392752** | **35.742%** | **70.785%** |
| R57: weight 2× | 2.356122 | **2.611956** | 3.399166 | 35.531% | 70.557% |
| R56: weight 4× | 2.359880 | **2.610042** | 3.404587 | 35.514% | 70.540% |

All four rows have unchanged auxiliary +2 CE 2.889248 and +3 CE 3.070085:
the successor correction is attached only to the next-byte output head.

## Controlled evidence and decision

- **R55 (1×):** overall CE advantage over the parent is +0.005523 nats/target,
  with within-arm corrected 95% interval [+0.000107, +0.012678]. Correct
  addresses beat wrong addresses by +0.005472 [+0.000131, +0.012540]. But the
  overall correct-successor advantage over deliberately rotated values is
  +0.000699 [−0.001032, +0.002292], and non-space CE against the parent is
  +0.000233 [−0.002697, +0.004078]. The space-byte gain dominates.
- **R56 (4×):** non-space CE advantage is +0.007337 over the parent,
  [+0.004018, +0.010273]; +0.007197 over wrong address
  [+0.003946, +0.010123]; and +0.004843 over wrong successors
  [+0.001852, +0.007423]. However overall CE is *worse* by 0.002681
  nats/target (corrected interval for improvement crosses zero), exposing
  a space/content tradeoff.
- **R57 (2×):** both overall and content CE improve on point estimates.
  Non-space CE improves +0.005423 over the parent
  [+0.002736, +0.007966]; overall CE advantage is only +0.001077
  [−0.002273, +0.004962]. Against wrong-successor values, overall CE favors
  the control on the point estimate. This arm also fails the global gate.

**Decision: no promotion, no independent TEST opening.** The inherited
step-2,993 model remains the protected baseline; all R55–R57 step-3,089
outputs remain unpromoted alternatives. Two things are now mechanistically
supported on real books: the gate can learn from historical byte successors,
and correct address/value content can reduce non-space loss. They have not
yet shown simultaneous statistically reliable overall and content gains under
both negative controls.

## Recovery, reproducibility and next architecture target

The three full JSON reports are under `results/FA2_R55*`, `FA2_R56*`, and
`FA2_R57*`. The reusable TRAIN-only tool is
`scripts/train_r55_pair_successor.py`; `tests/test_r55_pair_successor_train.py`
checks migration, unified optimizer retention, reversible interventions and
training only the new parameter. It does not use TEST for hyperparameter
selection or write over the original checkpoint. Source-level CI is separate
from genuine book-backed CPU evaluation.

To reproduce a source run from the exact parent and verified corpus:

```bash
cd poincare_mirror
python scripts/train_r55_pair_successor.py \
  --checkpoint /path/to/FA2_Geometry_Warmup_step2993_UNPROMOTED.pt \
  --data-dir /path/to/gutenberg38 \
  --output-checkpoint /path/to/new_unpromoted_candidate.pt \
  --output-report /path/to/new_DEV_report.json \
  --context 64 --train-windows-per-book 8 --dev-windows-per-book 8 \
  --batch 2 --updates 96 --lr 0.02 --nonspace-weight 1 \
  --bootstrap 2000 --seed 2026 --threads 2
```

The next engineering question is whether an **output-aware trust correction**
can preserve space prediction while selectively using verified non-space
historical evidence. That path must zero-initialize its new capacity, migrate
all inherited optimizer history, and demonstrate a robust correct-value
advantage. Long-context regression and independent sealed TEST follow only
after DEV is convincingly positive.
