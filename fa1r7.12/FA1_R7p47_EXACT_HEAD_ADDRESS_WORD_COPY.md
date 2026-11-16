# FA1 R7.46–R7.47 — exact head-address attribution and causal word copying

**Status:** unpromoted FA1 R7 research on local `master` only; source push
requires explicit authorization. One inherited network, one optimizer, and
all historical weights/moments protected. The accepted step-14,416 R7.13
lineage remains untouched. No independent sealed TEST books were opened.

## R7.46 diagnosis: what actually contributes?

The R7.45 value read projects the final descended Diamond point message
against embeddings of *observed pre-target word IDs*. R7.46 evaluated a
matched factorial ablation on 2,048 positions from all 15 NaNROUTER DEV
books, using the genuine trained 26-group R7.45 checkpoint (SHA-256
`7a9d35fea47cc74a2f8baa99fae49bda7e2e9c998f5ea7fca9769d2d09cdb48c`).
Every factor preserves the current target, underlying inherited R7 predictor,
and original observed candidate word values.

| Control | True minus control, R7.45 incremental benefit | Whole-book 95% interval |
|---|---:|---:|
| Swap only exact 256-point vertex addresses | −0.0000093 | [−0.0000373,+0.0000141] |
| Reorder the same eight active points | +0.0000063 | [−0.0000014,+0.0000163] |
| Swap only byte and sentence histories | +0.0000014 | [−0.0000092,+0.0000109] |
| Swap word, byte and sentence towers | **+0.0005722** | **[+0.0001548,+0.0011172]** |
| Zero the apex | +0.0001128 | [+0.0000086,+0.0002737] |

Thus the model extracts causal lexical-history information, but exact
geometric address identity was not demonstrated to be responsible. R7.46
was **diagnostic only**; it created no weights or checkpoint.

The true R7.45 incremental gain on this separately sampled DEV set was
+0.0004720 nats/word with a whole-book interval crossing zero. In the same
probe, 775/2,048 targets appeared among prior eligible candidate words:
they gained +0.0012474 nats/word on average. The other 1,273 positions
received precisely zero incremental R7.45 benefit. This reflects the
bounded *copy* nature of the geometric read, not general semantic prediction.

## R7.47 architecture: observed word values at real 256-head addresses

Every accepted R7 vocabulary word already has **two immutable historical
head addresses** in the accepted `token_heads.u16` table. R7.47 uses that
mapping in the same model:

1. Retrieve exactly the *pre-target* 32 recent and up to 12 near/far modal
   candidate word IDs; keep their original causal eligibility masks.
2. Map each candidate to its two accepted 256-head addresses. Compare them
   with the eight active history-derived point addresses.
3. Construct five bounded features: first-head match, second-head match,
   both-head pair match, aligned (S+) pair order, and crossed (S−) pair order.
4. Learn five small attention biases; reweight the already normalized R7.45
   candidate attention to obtain a normalized `Q_head(word)` distribution.
5. Add **two exact-zero bounded gates**: one mixes `Q_head` into the existing
   R7.45 word-copy distribution (at most 50% of its branch), and the other
   adds at most 12.5% total extra copy probability. The latter is controlled
   by target-free *observed head-match prevalence*, not by gold target hits.
6. The full inherited R7.45 word distribution remains in the same convex
   mixture. All copied words were historically observed; any unobserved
   vocabulary word receives zero `Q_head` probability.

At zero migration, both new gates are closed, giving **bit-identical** R7.45
scores. At nonzero gates, the normalized probabilities are nonnegative and
the final copy mixture remains bounded. The five new tensors are appended
as **optimizer group 27**, retaining all prior groups and moments, rather
than creating another network or optimizer.

Training uses the full normalized word NLL plus a counterfactual margin on
the **incremental R7.47 benefit under point-address swaps only**. The
frozen R7.45 predictor is subtracted on both control paths, so inherited
pre-existing address sensitivity cannot masquerade as a new contribution.

## Checkpoint-backed CPU pilot and larger DEV comparison

Source: recovered unpromoted R7.45 26-group checkpoint SHA-256
`7a9d35fea47cc74a2f8baa99fae49bda7e2e9c998f5ea7fca9769d2d09cdb48c`.
R7.47 trained **192 updates** on 2,048 TRAIN positions with minibatches of
16 and two CPU threads. The 512-position, 15-book development selection
chose update 192, but its strict geometric gate failed. The selected
checkpoint SHA-256 is
`5f221aa492446744724c4bb527e3d62d0ac9ef8f52009f3255b91bcf8ab7c96a`.

A separately sampled 2,048-position comparison on all 15 DEV books,
using seed 811 and independent reload of both checkpoints, found:

| Statistic | R7.45 parent | R7.47 candidate |
|---|---:|---:|
| Full normalized next-word NLL ↓ | 5.404824 | **5.399044** |
| R7.47 gain (nats/word) ↑ | — | **+0.005780** |
| Whole-book 95% bootstrap interval for gain | — | **[+0.002449,+0.009162]** |
| Correct vs wrong-book word history advantage ↑ | — | **+0.021460** |
| Whole-book 95% interval for correct-history advantage | — | **[+0.012680,+0.030695]** |
| Mean overall copy weight | — | 0.011644 |

Attribution gates on *incremental R7.47* benefit:

| Causal control | Mean true minus control | Whole-book 95% interval |
|---|---:|---:|
| Exact 256-point addresses only | −0.0000004 | [−0.0000915,+0.0000860] |
| Disable pair-down pathway | +0.0006784 | [−0.0000919,+0.0015320] |
| Disable apex | +0.0001035 | [−0.0001317,+0.0003585] |

The **word-history benefit is positive** on this DEV protocol. It remains
unproven that the gain depends on the intended pair/head geometry. The
strict DEV gate fails, and there is no promotion or frontier-performance claim.

## Restoration and integrity

- 448 inherited R7.45 model tensors are **bit-identical** in the saved R7.47
  checkpoint; exactly six new state-dict tensors were appended (five new
  parameters plus one static accepted vocabulary-head mapping).
- All **26 inherited optimizer parameter groups** and all **427 existing
  AdamW states** are bit-identical. Historical legacy optimizer state is
  also identical. The successor has one optimizer with **27 groups**.
- Exact save/reload of all tensors succeeded; 113/113 current source tests
  passed, including signed address orientation, normalized Q, target-free
  inputs, zero-gate parity, and donor controls.
- The SHA-disjoint 270-TRAIN/15-DEV/15-reserved-TEST short-story split matches
  the recovered release manifests. Earlier R7 ancestor corpus overlap is
  unknown, so these figures are not independent claims of broad test-set
  generalization. Repeated investigation of the same DEV books also makes
  the bootstrap intervals descriptive, not confirmatory discovery-adjusted
  evidence.
- Full-mixture next-word top-1/top-5 and raw uninterrupted UTF-8 bits/byte
  remain unavailable with the inherited target-conditioned retrieval path.
  First-byte and conditional spelling metrics are **not** substitutes.

**Next technical bottleneck:** strengthen the actual C(256,2) head-address
coupling with trainable pair-value information and test it under address-only,
same-point reordered, and unrelated-book controls before any TEST run.
