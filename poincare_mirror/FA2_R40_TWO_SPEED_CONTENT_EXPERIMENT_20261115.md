# FA2 R40 — matched content-weighted and two-speed Adam continuation

**2026-11-15 · `master` source experiment · no checkpoint promotion**

The single inherited `MirrorLM` and **all 194 Adam state records** were
continued from a recoverable, SHA-256-verified **unpromoted** geometric
checkpoint at step **2,993**. The original accepted R30 step-2,801 fallback
was neither modified nor replaced. Neither this round nor R39 recovered the
separate R38 step-4,057 successor-memory checkpoint.

## Reproducibility and protection

* Parent: `FA2_Geometry_Warmup_step2993_UNPROMOTED.pt`, SHA-256
  `74478d2f77b3c2da7978458f3660dffe8aad788938ce4a7ab6eae7cff2de0817`.
* Corpus: exactly 38 independently allocated books (12 TRAIN, 12 DEV, 14
  unopened TEST); `split_manifest.json` SHA-256
  `a7bb4406b1606def1be2aefdce0c2a6962999459eced6bc19b460d7ce55e96fc`.
* Models: original 23,472,023-parameter 192D/6-block/16-head architecture;
  no new replacement backbone, vocabulary, optimizer or model dimension.
* Each candidate resumed from an **independent byte-for-byte copy** of that
  exact checkpoint and ran precisely 16 Adam updates (2,993 to 3,009),
  train context 128, batch 1, CPU 2 threads, same saved sampler/RNG.
* Compared at identical positions on 12 DEV books, eight 128-byte windows
  per book (12,288 predictions/model), 2,000 paired whole-book bootstraps.
  All CE values are unweighted during evaluation, in nats/byte.

## Independent-book results (context 128)

| Metric | Parent | 1.50 weight, uniform LR 1e-5 | 1.50 weight, two-speed | 1.15 weight, uniform LR 1e-5 |
|:--|--:|--:|--:|--:|
| +1 CE ↓ | 2.383392 | 2.382705 | 2.383438 | **2.381916** |
| +1 BPB ↓ | 3.43851 | 3.43752 | 3.43858 | **3.43638** |
| +1 top-1 ↑ | **35.7015%** | 35.5062% | 35.3434% | 35.5876% |
| +1 top-5 ↑ | **69.8812%** | 69.5638% | 69.5231% | 69.7998% |
| Non-space +1 CE ↓ | 2.650380 | 2.634675 | **2.627541** | 2.641309 |
| Non-space +1 top-1 ↑ | 26.8574% | **27.3062%** | 27.2733% | 27.1186% |
| +2 CE ↓ | 2.926668 | 2.926266 | 2.926055 | **2.925812** |
| +3 CE ↓ | 3.106511 | 3.104960 | 3.105809 | **3.104948** |

Every candidate improved non-space CE, but **none passed** the complete
development gate. The 1.50-weight uniform candidate improved non-space CE
by 0.015705 nats/byte (95% whole-book CI `[+0.013900, +0.017587]`).
The two-speed candidate improved it by 0.022839 nats/byte
(`[+0.020174, +0.026421]`), but lost 0.3581 percentage points in overall
top-1. Milder 1.15 weighting gained 0.009070 nats/byte non-space
(`[+0.007146, +0.011156]`) and lost only 0.1139 points overall top-1.
These are training effects, **not demonstrated geometric-memory gains**.

## Actual geometric and long-context controls

* For the two-speed candidate, reciprocal feedback versus disabled **lost**
  0.000607 nats/byte on non-space CE; 95% CI
  `[-0.001486, +0.000261]`. No positive geometric-specific benefit.
* For the milder candidate the same control was -0.000024 nats/byte;
  95% CI `[-0.000661, +0.000800]`.
* Milder candidate evaluated at context **1,024**, one matched window/book
  (12,288 predictions/model): +1 CE 2.319299 → 2.319348;
  +2 CE 2.870761 → 2.871542; +3 CE 3.052670 → 3.052012;
  +1 top-1 35.0260% → 35.0505%, top-5 70.7438% → 70.7357%.
  The small +1/+2 long-context regressions preclude promotion.
* Sealed TEST predictions were not consulted during candidate selection.

## Architectural option: two-speed geometric AdamW

The opt-in flag `--inherited-step-scale 0.1` preserves the single
checkpoint, optimizer parameter group, raw Adam gradients, moment tensors,
and bias-correction history. **After** a normal AdamW step, it attenuates
the completed parameter displacement of inherited language parameters to
10%, while existing `reciprocal`, `hierarchy`, and `symbiotic` geometry
receive 100% of the displacement. This is a reversible **training protocol**,
not a new head or separate model. Default `1.0` leaves old behavior
unchanged. It did not pass the DEV gate and is therefore only an experimental
training option.

The read-only lineage audit verified 194/194 state IDs and moment schemas,
193 advanced Adam counters, stable configuration/corpus and preserved
single-optimizer parameter group for the two-speed and milder candidates.

## Reproduction commands

Create a separate candidate directory with `latest.pt` copied from the
parent above. Never point training or the `--steps 0` command at the parent.

```bash
python poincare_frontier.py train --data-dir DATA --run-dir CANDIDATE \
  --steps 16 --batch 1 --train-context 128 --eval-context 128 \
  --nonspace-loss-weight 1.5 --inherited-step-scale 0.1 \
  --lr 0.0001 --eval-every 8 --save-every 8 --cpu --cpu-threads 2

python scripts/audit_r40_lineage.py --before PARENT.pt --after CANDIDATE/latest.pt
python scripts/compare_geometry_books.py --before PARENT.pt \
  --after CANDIDATE/latest.pt --data-dir DATA --split valid \
  --context 128 --windows-per-book 8 --bootstrap 2000 --seed 2026 --threads 2
python -m unittest discover -s tests -q
```

## Next architecture gate

Train a causal, input-conditioned trust mechanism for geometric feedback,
initialized to preserve the exact inherited predictions. Evaluate the same
matched book windows with feedback enabled/disabled. Only broaden the
geometry if the *isolated* gain is positive with a whole-book lower CI above
zero and overall CE/top-1/top-5, future horizons, and aligned longer context
also pass. The unavailable R38 successor-memory results must not be
attributed to this checkpoint or topology.
