# FA2 R42 — Train-only counterfactual geometric-trust supervision

**2026-11-15 · experimental source on `master` · no checkpoint promotion**

## What changed

The R41 eight-weight causal reciprocal trust head was operational but had not
shown positive predictive contribution. R42 provides direct *TRAIN-only*
supervision of that same head. In each training microbatch:

1. The single inherited model predicts the causal next byte with the existing
   reciprocal pair / triangle / apex bridge enabled.
2. The **same current parameters** predict the same bytes with only that
   reciprocal bridge disabled, without tracking gradients.
3. Per-token next-byte cross-entropy labels whether the bridge helped. A
   detached-feature binary-classifier loss teaches the existing eight scalars
   with soft targets 0.9 (helped) and 0.1 (hurt), weighted by the opt-in
   `--trust-counterfactual-weight`.
4. The normal byte-likelihood and +2/+3 training objectives remain intact.
   Neither future target nor counterfactual result is an inference input.

The default auxiliary weight **zero** preserves R41's behavior. The training
path neither adds a parameter nor an optimizer. The source checkpoint is
restored with the original one-group AdamW history. New diagnostic fields
`train_trust_aux_bce` and `train_reciprocal_better_fraction` are emitted only
when the auxiliary term is enabled. The counterfactual uses temporary
`reciprocal.enabled=False` with `try/finally` restoration.

## Reproducibility and lineage

* Protected parent: FA2 geometry-warmup step **2,993**; SHA-256
  `74478d2f77b3c2da7978458f3660dffe8aad788938ce4a7ab6eae7cff2de0817`.
* Candidate: separate step **3,025** checkpoint; SHA-256
  `5ef9ff888931dea776e6ae5fd6cc401ec88803c0f5a07edd38ab1a16a6214ac4`.
* Same 38-book NaNROUTER whole-book allocation: 12 TRAIN / 12 DEV / 14
  unopened TEST; manifest SHA-256
  `a7bb4406b1606def1be2aefdce0c2a6962999459eced6bc19b460d7ce55e96fc`.
* 32 CPU AdamW updates; batch 1; causal training context 128; 2 threads;
  learning rate 1e-4; inherited displacement scale 0.05; non-space target
  weight 1.15; counterfactual auxiliary weight **0.2**.
* Audit: **194/194 inherited Adam states retained**, 193 advanced, 195
  candidate states including the eight-weight R41 head; one optimizer group.
* Local unittest: **39 tests passed**, including classifier sign/gradient,
  no LM-backbone gradient from the auxiliary path, shape guards, R41 exact
  checkpoint migration, and causal-prefix invariance.

## Real matched independent-DEV result

12 whole DEV books × 8 fixed windows/book × 128 positions = **12,288
matched next-byte predictions per model**; 2,000 paired whole-book
bootstrap resamples. These are DEV results, not TEST results.

| Metric (down/up) | Protected parent | R42 candidate |
|:--|--:|--:|
| +1 CE ↓ | 2.383392 | 2.382095 |
| +1 bits/byte ↓ | 3.43851 | 3.43664 |
| +1 top-1 ↑ | **35.7015%** | 35.4574% |
| +1 top-5 ↑ | **69.8812%** | 69.6370% |
| Non-space +1 CE ↓ | 2.650380 | **2.643568** |
| Non-space top-1 ↑ | **26.8574%** | 26.8360% |
| Non-space top-5 ↑ | 65.6387% | **65.7106%** |
| +2 CE ↓ | 2.926668 | 2.926113 |
| +3 CE ↓ | 3.106511 | **3.103181** |

Parent → candidate +1 CE gain **+0.001297 nats/byte** (95% book-bootstrap
CI [-0.002546,+0.004593]); non-space CE gain **+0.006812**
([+0.004468,+0.009175]); +3 CE gain **+0.003330**
([+0.001784,+0.004784]). Top-1 and top-5 decline.

### The decisive causal ablations

* Trust ON vs *identical weights with eight trust scalars neutralized to
  zero*: non-space CE advantage **+0.000012 nats/byte**, 95% bootstrap CI
  **[-0.000005,+0.000043]**, only 2/12 books improve. No significant
  improvement attributable to the new trust head.
* Full reciprocal bridge vs *identical weights with bridge disabled*:
  non-space CE gain **+0.000080**, 95% CI
  **[-0.001280,+0.001428]**. The bridge itself is not independently useful
  on this measurement.
* Candidate's eight trust weights have a different magnitude and direction
  from the R41 32-update matched ordinary run, so the auxiliary objective is
  **active**; its changed trust decisions have not yielded a measurable
  predictive gain.
* Direct R41 32-update **no-auxiliary vs R42 32-update auxiliary** comparison,
  with identical parent, RNG, optimizer schedule, and DEV windows: +1 CE
  gain **0.000000** (CI [-0.000021,+0.000021]); non-space CE gain
  **+0.000003** (CI [-0.000019,+0.000025]); +1 top-1 and top-5 are
  **identical to six decimals**. This specifically rejects a claim that
  counterfactual trust training has improved language accuracy.

**Decision: REJECT for promotion.** The independent DEV gate fails, and
independent long-context and sealed TEST gates have not been invoked. Do not
confuse improvement in non-space CE of *all weights together* with benefit
caused specifically by geometric trust.

## Reproduce, without overwriting the parent

Copy the canonical parent checkpoint to a fresh candidate run directory.

```bash
python poincare_frontier.py train --data-dir DATA --run-dir CANDIDATE \
  --steps 32 --batch 1 --train-context 128 --eval-context 128 \
  --reciprocal-content-trust --inherited-step-scale 0.05 \
  --nonspace-loss-weight 1.15 --lr 0.0001 \
  --trust-counterfactual-weight 0.2 --save-every 16 \
  --eval-every 16 --cpu --cpu-threads 2

python scripts/audit_r41_migration.py --before PARENT.pt \
  --after CANDIDATE/latest.pt
python scripts/compare_geometry_books.py --before PARENT.pt \
  --after CANDIDATE/latest.pt --data-dir DATA --split valid \
  --context 128 --windows-per-book 8 --bootstrap 2000 --threads 2
python -m unittest discover -s tests -q
```

**Next hypothesis:** the limiting factor is now the extremely weak
reciprocal bridge itself, not the trust classifier. A future experiment
should isolate bridge-specific learning from inherited trunk updates and
measure whether stronger, causally useful pair/triangle retrieval improves
the *same* held-out bytes under an honest bridge-disabled ablation.
