# FA2 R44 — Causal geometric residual-correction supervision

**2026-11-15 · experiment under `poincare_mirror` on `master` · no promotion**

## Research question

R41/R42 content trust learned an almost neutral gate, and R43 showed that
simply amplifying reciprocal feedback on DEV books substantially damaged
likelihood. R44 tests whether the problem is *the information carried by
the geometric representation*.

R44 adds no neural parameters. During training only, the **existing**
five-scale / C(256,2) mirror-pair / C(256,3) triangle / 256-apex geometry
exposes its last causal hidden residual **before** the tiny reciprocal gate.
The same model is also evaluated with reciprocal feedback disabled, with
no gradient tracked. The detached disabled logits plus a projection of
the geometric residual through the existing **tied vocabulary** predict
the current TRAIN next-byte target:

`correction_logits(t) = detached_off_logits(t) +
    logit_scale * linear(ungated_geometry(t), detached_tied_byte_weights)`.

An optional content-weighted CE on these logits contributes to training;
it is never used in inference or validation. No future-target bytes enter
causal geometry, retrieval, or the model input. The new flags default to
off, preserving exact R43 behavior:

* `--reciprocal-residual-aux-weight` (default 0)
* `--reciprocal-residual-logit-scale` (default 8)

The existing `--geometry-only` mode makes this an informative isolation
test: all inherited trunk weights and their Adam moments remain frozen,
while the **single** optimizer trains attached geometry. No replacement
model, vocabulary, optimizer, or branch was created.

## Matched CPU training

Both arms began from the independently recoverable FA2 geometry-warmup
checkpoint at step **2,993**; SHA-256
`74478d2f77b3c2da7978458f3660dffe8aad788938ce4a7ab6eae7cff2de0817`.

* Original model: 192-dimensional, six blocks, 4,096 maximum context,
  23,472,023 parameters; R41 trust expansion adds eight scalars.
* 12 TRAIN books, 12 whole DEV books, 14 unopened TEST books; corpus
  split-manifest SHA-256
  `a7bb4406b1606def1be2aefdce0c2a6962999459eced6bc19b460d7ce55e96fc`.
* Two 32-update geometry-only AdamW continuations through step **3,025**;
  batch 1, training context 128, two CPU threads, LR 1e-4, train-only
  non-space target weight 1.15, no checkpoint or optimizer reset.
* Control: no residual auxiliary loss. Candidate: auxiliary weight **0.25**,
  residual logit scale **8**. The first concurrent experiment exceeded
  container memory; the candidate was rerun **alone from the original
  parent** and completed all 32 steps with the same RNG and settings.

## Exact lineage audit

Both arms passed the R44 frozen-trunk audit:

| Identity / audit | Control | R44 auxiliary |
|:--|--:|--:|
| Inherited Adam states retained | 194/194 | 194/194 |
| Frozen trunk parameter sets unchanged | 63 | 63 |
| Frozen trunk Adam state sets bitwise identical | 63 | 63 |
| Geometric parameter sets changed | 131 | 131 |
| Geometric Adam state sets advanced | 131 | 131 |
| Trainable/frozen parameter elements | 2,167,039 / 21,304,992 | 2,167,039 / 21,304,992 |

The two candidate checkpoint SHA-256 digests are:

* Geometry-only control:
  `e20a0b9d74547bab97137bb13cf52cafd170eb2729bcd1fab3524e63b99ef3e7`.
* R44 residual objective:
  `fda6558b290924d0dfa93c3d270294ad37b930787c167128b6ca1554b5a592cb`.

## 12-book matched DEV assessment

Each arm was evaluated at exactly the same **12 independent DEV books ×
8 windows/book × 128 tokens = 12,288 positions**, with 2,000 whole-book
bootstrap resamples. No TEST books were used for hyperparameter choices.

| Metric | Geometry-only control | R44 residual objective |
|:--|--:|--:|
| +1 CE ↓ | **2.383983** | 2.384695 |
| +1 bits/byte ↓ | **3.43936** | 3.44039 |
| +1 top-1 ↑ | 35.6771% | **35.7178%** |
| +1 top-5 ↑ | 69.7835% | **69.7917%** |
| Non-space +1 CE ↓ | **2.649442** | 2.649764 |
| Non-space +1 top-1 ↑ | 26.8920% | **26.9601%** |
| +2 CE ↓ | **2.927892** | 2.928017 |
| +3 CE ↓ | 3.105278 | **3.105153** |

The auxiliary objective's matched CE *gain* is **−0.000712 nats/byte**
with 95% bootstrap CI **[−0.001741,+0.000189]**. Non-space CE gain is
**−0.000322** with CI **[−0.000954,+0.000326]**. The tiny top-1 gain
is **+0.000407** in fractional accuracy, CI
**[−0.000490,+0.001139]**, not statistically established.

More importantly, the R44 candidate's **reciprocal bridge ON vs disabled**
non-space CE advantage is **−0.000948 nats/byte**, 95% CI
**[−0.001504,−0.000371]**. This is significant *harm from enabling the
current feedback*, not evidence of geometric memory improvement. Its
trust ON vs neutral non-space CE gain was just **+0.000002**, CI
**[−0.000002,+0.000009]**. The DEV promotion gate is **false**.

An additional matched **protected parent → R44** comparison gives +1 CE
**2.383392 → 2.384695** (gain −0.001303, CI
[−0.003771,+0.000850]); +2 CE **2.926668 → 2.928017**; +3 CE
**3.106511 → 3.105153** (gain +0.001358, CI
[+0.000033,+0.002454]). The +3 improvement is interesting but cannot
justify promotion against degrading next-byte and +2 likelihood or
weak feedback ablations.

The next research step must improve the *causal content selection* at
pair/triangle/apex addresses themselves, including correct-history vs
wrong-history controls. A bigger gate, wider model, or longer training run
alone is not supported by these results.

## Reproduce

Copy the protected step-2,993 source checkpoint into separate experiment
directories before executing. Do **not** train in place against the parent.

```bash
python poincare_frontier.py train --data-dir DATA --run-dir CONTROL \
  --steps 32 --batch 1 --train-context 128 --eval-context 128 \
  --reciprocal-content-trust --geometry-only \
  --nonspace-loss-weight 1.15 --lr 0.0001 \
  --save-every 16 --eval-every 16 --cpu --cpu-threads 2

python poincare_frontier.py train --data-dir DATA --run-dir AUX \
  --steps 32 --batch 1 --train-context 128 --eval-context 128 \
  --reciprocal-content-trust --geometry-only \
  --nonspace-loss-weight 1.15 --lr 0.0001 \
  --reciprocal-residual-aux-weight 0.25 \
  --reciprocal-residual-logit-scale 8 \
  --save-every 16 --eval-every 16 --cpu --cpu-threads 2

python scripts/audit_r44_geometry_only.py --before PARENT.pt \
  --after AUX/latest.pt --requires-aux
python scripts/compare_geometry_books.py --before CONTROL/latest.pt \
  --after AUX/latest.pt --data-dir DATA --split valid \
  --context 128 --windows-per-book 8 --bootstrap 2000 --threads 2
python -m unittest discover -s tests -q
```

**Decision: R44 rejected for checkpoint promotion.** Preserve the parent,
retain this opt-in experimental source and exact DEV data, and keep the
independent TEST split unopened.
