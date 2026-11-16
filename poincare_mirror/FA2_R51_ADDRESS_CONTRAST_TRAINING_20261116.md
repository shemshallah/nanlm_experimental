# FA2 R51 — Bounded correct-vs-shuffled-address training, measured and rejected

Date: 2026-11-16 UTC. Track: **FA2 Poincaré Mirror**; neither FA1 nor
Terminal-RSI is used. GitHub branch: `master` only.

## Purpose and exact inherited lineage

R50 established that the protected step-2,993 FA2 model is nearly insensitive
to exact C(256,2)/triangle-hash/incidence addressing. R51 tests whether a
direct contrastive training loss can make exact geometric addresses
predictively meaningful without introducing a new model, tokenizer head,
optimizer, or alternative parameter branch.

Protected input: SHA-256
`74478d2f77b3c2da7978458f3660dffe8aad788938ce4a7ab6eae7cff2de0817`.

The new CLI options in the existing `poincare_frontier.py train` command are
disabled by default and require `--geometry-only`. No behavior change occurs
for existing training jobs unless explicitly enabled.

For each causal TRAIN-only byte window, the existing language model computes
two next-byte distributions on **identical inputs and targets**:

- `exact`: standard point/pair/triangle/tetra/apex addresses.
- `scrambled`: reversible, independently generated pair-lookup, triangle-hash,
  and chain-consistent incidence relabeling.

The extra smooth-margin loss is

`T * softplus((m + CE(exact) - CE(scrambled)) / T)`

and is added at weight `w` to the inherited language objective. Both outputs
remain differentiable, so this is *not* merely ordinary CE upweighted by a
constant. Different TRAIN-only permutations are sampled at every microbatch.
Temporary incidence buffers are swapped **by reference**, preserving autograd
version counters of earlier causal forward graphs, and then fully restored.
This loss still risks learning to *degrade scrambled predictions* rather than
improving exact ones; the DEV gate explicitly rejects that failure mode.

## First bounded CPU pilot

| Property | Value |
|---|---|
| Checkpoint | Protected step 2,993 → unpromoted candidate step 3,001 |
| Updates | 8 |
| Training bytes/targets | 256 (`batch=1`, context=32, grad accumulation=1) |
| Weight / margin / temperature | 0.1 / 0.005 nats / 0.01 nats |
| Negative | `all_scrambled` with independent TRAIN-only permutation per step |
| Training data | 12 original book-separated NaNROUTER TRAIN books |
| DEV comparison | 12 separate DEV books × 8 × 64 = 6,144 matched prediction positions |
| Model parameters | 23,472,023 (inherited model) |
| Geometry trainable | 2,167,031 parameters |
| Inherited backbone | 21,304,992 parameters frozen |

The optimizer-lineage audit PASSED:

- 194/194 inherited Adam states retained.
- 63 inherited parameter sets and their Adam moments remain **bitwise equal**.
- 131 geometric parameter sets and corresponding Adam steps advanced.
- Effective architectural configuration and corpus manifest unchanged. The
  newer serializer explicitly writes the old implicit default
  `reciprocal_content_trust=false`; this is not an architecture change.

The unpromoted candidate SHA-256 is
`fab21bdef79a267de3322fa1e759eb4900a0efee27678ec2d0b4ca0aa579ae53`.

## Matched NaNROUTER DEV outcome

| Metric | Parent | R51 candidate | Candidate gain (positive=better) |
|---|---:|---:|---:|
| Next-byte CE, nats ↓ | 2.357199 | 2.355582 | +0.001617 |
| Next-byte top-1 ↑ | 35.5143% | 35.8561% | +0.3418 pp |
| Next-byte top-5 ↑ | 70.4915% | 70.6217% | +0.1302 pp |
| +2-byte CE ↓ | 2.889248 | 2.889215 | +0.000033 |
| +3-byte CE ↓ | 3.070085 | 3.071196 | **−0.001112** |
| Non-space next-byte CE ↓ | 2.617379 | 2.624022 | **−0.006644** |
| Non-space top-1 ↑ | 26.9378% | 26.8091% | −0.1287 pp |
| Non-space top-5 ↑ | 66.1771% | 66.1344% | −0.0427 pp |

The next-byte improvement has a descriptive 95% **book-cluster** bootstrap
interval of `[−0.000554, +0.003915]` nats/target and is not statistically
established. The non-space regression interval is
`[−0.008216, −0.004816]` nats/target; the +3 regression interval is
`[−0.002022, −0.000156]`. These are single-family descriptive intervals,
not a basis for promoting a checkpoint after repeatedly examining DEV.

Under the R50 address-integrity probe on the **same** candidate, `exact`
next-byte CE was 2.355582217. Shuffling all learned addresses gave
2.355571141 (a tiny improvement for shuffling), with the corrected paired
interval crossing zero. The newly trained model still does **not** require
its exact geometric addresses to predict well.

Next-byte bits per byte/EOS target: **3.400720 → 3.398387**. This should not be
misinterpreted as a meaningful low-BPB improvement when non-space content
likelihood and the third horizon both worsen.

Raw outputs:

- `results/FA2_R51_lineage_DEV_20261116.json`
- `results/FA2_R51_vs_parent_DEV_20261116.json`
- `results/FA2_R51_candidate_address_DEV_20261116.json`

## Verdict

**R51 candidate REJECTED**, no independent TEST opened. The protected
step-2,993 checkpoint and complete 194-state Adam lineage remain canonical.
The 8-update successor is retained only as a recoverable negative experiment.
This pilot is much too short to judge the larger-scale potential of address
contrastive training; its DEV and attribution gates nevertheless rule out
promoting this specific candidate.

The next architectural priority is to reduce the space/non-space imbalance
*and* demonstrate a correct-address advantage using held-out, independent
books before increasing the learning signal or training duration.
