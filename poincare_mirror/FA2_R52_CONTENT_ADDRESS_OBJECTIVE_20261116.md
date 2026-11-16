# FA2 R52 — Content-byte geometric address contrast (unpromoted)

Date: 2026-11-16 UTC. FA2 Poincaré Mirror only, on existing `master`.
FA1, M3, NaNLMind, and Terminal-RSI remain separate.

## Motivation and intervention

R51's contrastive TRAIN-only exact-versus-scrambled-address objective reduced
aggregate next-byte loss but improved spaces while making **non-space CE worse**.
R52 targets that specific failure without replacing the inherited model or
Adam optimizer. The opt-in `--address-contrast-content-only` masks byte 32
(space) and token 256 (EOS) **only within the geometric ranking loss**.
The inherited language CE is still computed for all targets, with
`--nonspace-loss-weight 1.5` in this pilot. The two contrastive next-byte
paths use the same causal input and TRAIN-only labels, and both receive
gradients; geometric address shuffles remain reversible and target-free.

The original R51 behavior is unchanged when the new flag is omitted. For an
all-space/EOS microbatch the auxiliary loss is a differentiable zero. This
does not discard document boundary information from the main LM objective.

## Fixed experimental protocol

- Protected parent: step **2,993**, SHA-256
  `74478d2f77b3c2da7978458f3660dffe8aad788938ce4a7ab6eae7cff2de0817`.
- Existing 23,472,023-parameter model and single unified Adam optimizer.
- Training: **16** updates, context 32, batch 1, accumulation 1;
  **512 TRAIN target positions**, 12 book-separated NaNROUTER TRAIN books.
- Geometry-only updates; 2,167,031 geometric parameters trainable,
  21,304,992 inherited backbone parameters frozen.
- Exact-address contrast weight 0.05, margin 0.005 nats, temperature
  0.01 nats, fresh deterministic `all_scrambled` permutation per microbatch.
- Evaluation: **12 independent DEV books × 8 matched windows × 64**
  = 6,144 positions per model, seed 2026, 2,000 whole-book bootstrap samples.
- No independent TEST, checkpoint promotion, or replacement model.

## Matched DEV results (lower CE is better)

| Metric | Protected step 2,993 | R52 step 3,009 | Candidate gain |
|---|---:|---:|---:|
| Next-byte CE, nats | 2.357199 | 2.355772 | +0.001427 |
| Next-byte BPB/EOS target | 3.400720 | 3.398661 | +0.002059 |
| +2 CE, nats | 2.889248 | 2.889950 | **−0.000702** |
| +3 CE, nats | 3.070085 | 3.071747 | **−0.001662** |
| Non-space next-byte CE, nats | 2.617379 | 2.618473 | **−0.001095** |
| Next-byte top-1 | 35.5143% | 35.7259% | +0.2116 pp |
| Next-byte top-5 | 70.4915% | 70.6217% | +0.1302 pp |

The whole-book 95% descriptive CI for next-byte CE gain is
`[-0.000635, +0.003795]` nats/target (crosses zero). For +3 loss gain
the CI is `[-0.002777, -0.000538]`, entirely adverse. Non-space CE gain
CI is `[-0.003379, +0.001275]`, also crossing zero. The next-byte
model throughput was 1,091.4 scored positions/s versus parent 994.9 on
this CPU measurement; this is noisy evaluation timing, not a speed claim.

## Lineage and gate

Lineage audit passed: **194/194 inherited Adam states**; all 63 frozen
backbone parameter sets and moments bitwise identical; all 131 geometric
parameter sets changed and their optimizer steps advanced. The architecture
configuration and book-split manifest are unchanged.

**Candidate rejected for promotion.** Neither aggregate next-byte gain nor
content gain is established; longer-horizon regressions violate the promotion
gate. The protected step-2,993 parent remains the reference. A 512-position
pilot is insufficient to rule out longer-horizon research hypotheses, but
cannot justify scaling this configuration as though it succeeded.

## Address attribution and reproducibility

The completed address-attribution probe compared exact, scrambled pairs,
scrambled triangle addresses, chain-consistent scrambled incidence, all
scrambled, and disabled reciprocal emission on the *same* checkpoint and
the same 6,144 DEV positions:

| Addressing intervention | Next-byte CE, nats |
|---|---:|
| Exact | 2.355772336 |
| Pair addresses shuffled | 2.355772316 |
| Triangle addresses shuffled | 2.355772614 |
| Incidence relabeled | 2.355772724 |
| All shuffled | 2.355772694 |
| Reciprocal disabled | 2.356141190 |

The exact-vs-all-shuffled difference is only **0.000000358 nats/target**;
its Bonferroni-corrected whole-book CI crosses zero. There is **no
demonstrated predictive use of exact geometric addresses**. The model
recovers bitwise-identical exact predictions after all interventions, so
the test did not leave altered buffers or learned tensors behind. Removing
the entire reciprocal emission is somewhat worse on average, but its
corrected interval also crosses zero: presence of a geometric module is not
evidence that its exact topology is doing useful predictive work.

Raw DEV comparison: `results/FA2_R52_vs_parent_DEV_20261116.json`.
State/moment audit: `results/FA2_R52_lineage_DEV_20261116.json`.
Local source regression tests: **15/15 PASS**, including R51 backwards
compatibility and R52 masking/gradient checks. This R52 experiment is not a
frontier-level language-model result.
