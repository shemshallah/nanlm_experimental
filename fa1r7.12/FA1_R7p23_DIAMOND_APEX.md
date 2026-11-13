# FA1 R7.23 — Full C(256,3) Diamond / unified R7

## Scope and architecture

One R7 lineage on `master`, retaining the trained 96-dimensional R7 trunk, 256 learned H points, 32,640 inherited P/M directed mirror pairs, near 8K/32K and far 128K retrieval, and both 128K and 1M semantic readers (previously closed). No FA2 mixing.

Upward: 256 points → 32,640 exact lexicographic pairs → 2,763,520 distinct combinadic triples. Downward: triples → their three incident pairs → participating points → the inherited R7 hierarchical decoder. A 256-specialist learned meta-router judges the topological apex.

The meta-controller additionally learns whether the diamond tree likelihood may beat the frozen R7 baseline. It uses only target-free context at inference; ground-truth advantages supervise it during training. The apex consumes the inherited 92-dimensional target-free near/far retrieval statistics, not ground-truth target probabilities.

Each 1M-scale triple is individually addressable and has a learned three-float code, *not* a dense expert transformer. Only 8 context-selected points, 28 pairs and 56 triples activate per token on the CPU pilot. The original 256 point and 32,640 pair tensors are *shared*, not duplicated.

## Exact checkpoint roundtrip

- Parent R7.22 SHA-256: `eb0b7f0505f027095af23bc85ef00f2c036f3d2d5fb160ee0c45e60fb02e3455`.
- Original accepted step: **14,416**.
- Five inherited optimizer histories: **5/5, bitwise unchanged**.
- All inherited model tensors: **bitwise unchanged**.
- R7.23 identity checkpoint SHA-256: `e68b726d424d7ade61ff22483aa10e33ec0924e4a687a9100e1b912bb12736c7`.
- Checkpoint roundtrip: **passed** (new topology, learned tensors, optimizer histories, zero training updates).
- Identity gate is closed, so old R7.22 next-word NLL is preserved exactly. This checkpoint is a *recoverable experimental migration*, not a promoted performance candidate.

## Completed CPU pilots — historical development data only

| Run | Updates | Parent dev NLL | Final dev NLL | Δ NLL (lower better) | Correct vs unrelated | Selected | Seconds |
|---|---:|---:|---:|---:|---:|---|---:|
| direct192 | 192 | 4.928124 | 4.936072 | +0.007947 | -0.010541 | False | 41.1 |
| contrast288 | 288 | 5.171878 | 5.176274 | +0.004396 | +0.012963 | False | 135.5 |
| trust320 | 320 | 4.945808 | 4.948652 | +0.002844 | -0.002107 | False | 72.7 |
| memory384 | 384 | 5.065650 | 5.068772 | +0.003122 | +0.008690 | False | 104.1 |

**No measured promotion:** none satisfied development loss, correct-vs-unrelated causal history, apex ablation, and pair-down ablation simultaneously. On the direct arm, step 96 briefly gained ~0.00153 NLL, but correct history lost to unrelated history; the apparent improvement was rejected.

These historical NLL values are nats per vocabulary target. They are **not** measured byte-accurate BPB. No independent NaNROUTER book test has yet demonstrated an improvement. The CPU pilot does not imply frontier-scale language capabilities.

## Engineering tests

- Bijective address test of all 32,640 R7 pairs and all 2,763,520 apex triples: passed.
- Exact 56-to-28 triple/edge incidence per 8-point activation, with six triangles per active edge: passed.
- Trainable gradient path: triple codes, triple projection, meta specialists, pair-down, point-down, inherited retrieval feature pathway: passed.
- No target or future tokens are inputs to active vertices, routing, or meta-trust. Training labels enter losses only.
- The small random nonzero triple codes repair zero-gradient for the shared triple projection at initialization.
- GitHub R7.23 structural/inherited source-only workflow verified on `master`; this is not a checkpoint training run.

## Critical scientific conclusions and future criteria

The main blocker is **generalization and contextual causality**, not address-space capacity. Extra memory addresses and trainable specialists do not by themselves make an LLM frontier-quality. The apex direct language head overfits very small training sets and the learned reliability controller becomes overly trusting. The old R7.22 remains the accepted predictive baseline.

The next effective experiment is a curriculum on **book-disjoint NaNROUTER data**, with at least 1,024 held-out targets per independent document-cluster split, a frequency-only hard negative, shuffled/injected wrong histories, relative improvement versus accepted R7 on **all** splits, bootstrap confidence intervals by book, and a multi-hop downstream retrieval benchmark. Only if all gates pass should GPU training be considered for expanding active vertices or opening old trainable R7 tensors.

## Reproduction

Use `fa1r7.12/source/fa1_r723_seed.py` for the identity checkpoint migration and roundtrip verification. `fa1_r723_train.py` runs the staged supervised contrastive/meta-trust pilot. `test_fa1_r723_topology.py` verifies exact combinatorial topology and gradient coverage. The full original R7.13 tokenizer/corpus metadata and the R7.22 94.8 MB candidate file are required. No binary weights are embedded in the source ZIP.