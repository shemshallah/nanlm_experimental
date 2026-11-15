# FA2 R41 — Causal geometric-feedback trust gate

**2026-11-15 · experimental extension on `master` · no model promotion**

## Architecture

R41 adds exactly **eight trainable scalars** to the *existing* five-scale
reciprocal tokenizer / signed mirror-pair / triangle / apex feedback loop.
At each causal position the mean of the current five scale states produces
an input-conditioned factor

`trust(t) = 2 * sigmoid(w · mean_scales(t))`.

The range is `(0, 2)`; the new weight row is initialized to all zeros, so
`trust(t) = 1` exactly at migration. No future byte is used to make this
decision. The learned factor modulates the same point/pair/triangle/apex
reciprocal feedback and final token-scale residual; there is **no second
network, language head, hidden vocabulary, or optimizer**.

New opt-in flag: `--reciprocal-content-trust`. It requires an already
present reciprocal Diamond and may be enabled only once. Existing checkpoints
without this option retain their exact behavior and default config.
Function-preserving migration reindexes the **single** AdamW parameter group
by parameter name, keeping all inherited moments; the new row starts with
no Adam history until it first receives a gradient.

## Verified properties

* Deterministic synthetic micro/meta checkpoint migration: exact next-byte
  logits before and after opt-in expansion, including two-element batches;
  inherited Adam steps, first and second moments bit-for-bit preserved.
* Causal prefix property: changes to future bytes and embeddings do not
  change earlier trust values.
* Real 23,472,023 → 23,472,031-parameter migration from FA2 step 2,993.
  The two actual continuations retained **all 194 inherited Adam states**,
  with 193 inherited step counters advancing and **one new trust state**
  created; 195 total after training. Single optimizer group preserved.
* Parent SHA-256:
  `74478d2f77b3c2da7978458f3660dffe8aad788938ce4a7ab6eae7cff2de0817`.
* Same 38-book NaNROUTER allocation as R40, with 12 TRAIN / 12 DEV /
  14 sealed TEST. Whole-book split manifest SHA-256:
  `a7bb4406b1606def1be2aefdce0c2a6962999459eced6bc19b460d7ce55e96fc`.

## Real CPU training and DEV benchmark

One candidate ran **16** Adam updates with LR `1e-5`, non-space weight
`1.15`; another ran **32** updates with LR `1e-4`, inherited displacement
scale `0.05` and full-rate geometry, same weight `1.15`. Both independently
resumed from untouched step 2,993, without optimizer reset. All results
come from **12 whole DEV books × 8 × 128 matched positions = 12,288
predictions/model**, with 2,000 paired whole-book bootstraps.

| Measure ↓/↑ | Parent | R41 16 steps | R41 32 steps |
|:--|--:|--:|--:|
| +1 CE ↓ | 2.383392 | **2.381916** | 2.382095 |
| +1 bits/byte ↓ | 3.43851 | **3.43638** | 3.43664 |
| +1 top-1 ↑ | **35.7015%** | 35.5876% | 35.4574% |
| +1 top-5 ↑ | **69.8812%** | 69.7998% | 69.6370% |
| Non-space +1 CE ↓ | 2.650380 | **2.641310** | 2.643572 |
| Non-space top-1 ↑ | 26.8574% | **27.1186%** | 26.8360% |
| +2 CE ↓ | 2.926668 | **2.925812** | 2.926115 |
| +3 CE ↓ | 3.106511 | 3.104948 | **3.103174** |

The stronger run improved non-space CE by **0.006808 nats/byte** (book 95%
CI `[+0.004475, +0.009179]`); its +3 CE improved by **0.003337**
(`[+0.001788, +0.004787]`). These improvements are not attributable to
trust itself: the same existing language/geometry parameters were trained.
On the matched context-128 DEV pass with two CPU threads, measured
inference throughput was approximately **1,242 bytes/s** for the parent
and **1,257 bytes/s** for the candidate. The benchmark includes the R41
trust logic but not the additional ablation passes.

## Crucial R41-specific ablation

To isolate the contribution of the **new gate**, `compare_geometry_books.py`
now sets only `context_trust.weight = 0` (exactly neutral multiplier 1),
leaving every other trained parameter unchanged. It compares the same
causal positions with paired book-bootstrap confidence intervals.

* 16-step: gate-on non-space CE advantage rounded to **0.000000**;
  95% CI `[-0.000001, 0.000000]`.
* 32-step: gate-on non-space CE advantage **+0.000013 nats/byte**;
  95% CI `[-0.000002, +0.000042]`.
* 32-step: entire reciprocal bridge versus disabled non-space gain
  **+0.000086**, 95% CI `[-0.001286, +0.001463]`.

Thus the new eight-weight gate is **mechanically operational but not proven
predictively useful**. Overall top-1/top-5 still regress. **Both R41
checkpoints are rejected for promotion**; the sealed TEST split remains
unopened for this selection.

## Reproduce and continue safely

Copy the verified source checkpoint to a *separate, unpromoted* candidate
directory before invoking this command. Do not run training or `--steps 0`
against the parent checkpoint.

```bash
python poincare_frontier.py train --data-dir DATA --run-dir CANDIDATE \
  --steps 32 --batch 1 --train-context 128 --eval-context 128 \
  --reciprocal-content-trust --inherited-step-scale 0.05 \
  --nonspace-loss-weight 1.15 --lr 0.0001 \
  --save-every 16 --eval-every 16 --cpu --cpu-threads 2

python scripts/audit_r41_migration.py --before PARENT.pt \
  --after CANDIDATE/latest.pt
python scripts/compare_geometry_books.py --before PARENT.pt \
  --after CANDIDATE/latest.pt --data-dir DATA --split valid \
  --context 128 --windows-per-book 8 --bootstrap 2000 --threads 2
python -m unittest discover -s tests -q
```

The next worthwhile architecture research is **direct causal supervision
of when reciprocal memory helps**, measured against the *same-weight*
neutralized trust and reciprocal-off counterfactual. Avoid extending context,
dimension, head count, or training to millions of steps until the memory path
itself beats both ablations with a positive book-level lower CI, and all
overall/non-space accuracy and independent long-context gates pass.
