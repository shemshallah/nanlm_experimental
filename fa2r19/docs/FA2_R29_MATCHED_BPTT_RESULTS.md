# FA2 R29 — matched recurrent-horizon training and 132K causal replay

**Date:** 2026-11-12 US Pacific (2026-11-13 UTC). **Status:** measured exploratory experiments; **no R29 model or optimizer state was promoted**. The R27 recovery checkpoint remains immutable and unpromoted; the original R18 ancestor is still the formally accepted baseline. The R28 CPU16 adapter remains an independently restorable, **unpromoted** experimental candidate in the same unified FA2 model lineage.

## Why this round matters

R28 extended PAL's recurrent truncated backpropagation from two consecutive 12-word chunks to four. Its CPU16 pilot improved a small development sample, but a wider 4,482-byte, three-book sample reveals that its 0.000607 BPB improvement against R27 comes with a 0.001655 **regression** in non-space NLL. Therefore, simply maximizing aggregate bits-per-byte (BPB) risks optimizing easy whitespace at the cost of linguistic content.

This round separates **gradient horizon** from **number of tokens and optimizer steps**, then checks whether R28 actually survives causal 33K/65K/132K document-prefix replay.

## Unified state and integrity

- Original R27 unified candidate SHA-256: `0f4cdb8ce41b62472b6c8408296b496dfb3e801b388289453c0cdcc720f5b3f8`.
- R28 CPU16 497,239-byte adapter overlay SHA-256: `e29155c5bde6d33237f65d13b662e8ccebef3cd8adf89b075390993e20673fd2`.
- R27 inheritance: 12 optimizer groups, 144 Adam states, 12 original document stream snapshots.
- R28 experimental continuation: 13 optimizer groups, 153 Adam states; all inherited R27 tensors/moments retained bit-for-bit.
- **R29 candidate selection:** none. No R29 replacement or reset of the R28/R27 weights or optimizer was saved. The current recovery overlay is exactly R28 CPU16, not a silently promoted R29 model.

## Wider near-prefix baseline (before R29 training)

Three previously seen NaNROUTER development books (866, 970, 1444), 20 contiguous 12-word chunks per book, 4,482 bytes and 3,165 subword tokens:

| Metric (lower better) | R27 | R28 CPU16 |
|---|---:|---:|
| Bits/byte | 3.404849006 | 3.404242272 |
| Non-space NLL | **4.183219833** | 4.184874633 |

The R28 result is not a general improvement on this evaluation. It is an overall BPB gain with a non-space NLL regression.

## R29 equal-token, equal-update gradient horizon control

Two training books (240, 784), three development books (866, 970, 1444). Each of the **2-, 4- and 8-chunk** BPTT arms starts from exactly the same R28 CPU16 weights and 153 Adam states. Each update sees precisely **eight consecutive 12-word segments**, one optimizer step, and the same document selection and token positions. The only experimental variable is whether the recurrent graph is detached after 2, 4 or 8 segments. Each arm completed 16 updates and **4,890 training tokens** with identical inherited weights and training schedule. Evaluation after updates 4, 8, 12 and 16 used the same 4,482 development bytes. New gradients can pass through the inherited PAL recurrence, but only the existing R28 residual is updated.

| BPTT chunks | Best evaluated step | Best aggregate BPB | Corresponding non-space NLL | Passed dual-metric/per-book guard? |
|---|---:|---:|---:|---|
| Inherited R28 CPU16 | — | 3.404242272 | 4.184874633 | Reference only |
| 2 | 12 | **3.402197724** | 4.196224510 | **No** |
| 4 | 8 | 3.402979412 | 4.191440864 | **No** |
| 8 | 8 | 3.402944114 | 4.191264898 | **No** |

All attempted arms improved aggregate BPB at an intermediate checkpoint, but each worsened non-space NLL, with some per-book regressions. More gradient horizon did not reliably win under this matched protocol. **No checkpoint selected.**

## R29 weighted-loss repair experiment

The same R28 starting state was used again, with a lower learning rate of `0.0002` and a weighted per-token cross-entropy giving literal-space tokens weight `0.35` and all other targets weight `1.0`. Both 4- and 8-chunk arms received 12 updates and the same **3,772 training tokens** per arm. This round also required beating the R27 non-space benchmark, not only R28.

| BPTT chunks | Best evaluated step | Best aggregate BPB | Corresponding non-space NLL | Passed R27/R28 guards? |
|---|---:|---:|---:|---|
| 4 | 12 | 3.403963472 | 4.185101568 | **No** |
| 8 | 12 | 3.403990513 | 4.185035540 | **No** |

The lower-rate weighted objective reduced the severity of the non-space regression relative to the initial experiment, but did not eliminate it. **No checkpoint selected.** This is a narrow hyperparameter probe, not a proof that weighted training cannot work after a larger data budget.

## Long-prefix frozen replay: 33K / 65K / 132K

The model read **every prior subword** of each book through its GRU and its causal attention key/value history. There was no training, adaptation, model selection, or replacement of the prior state at evaluated positions. In the accelerated replay, historical logits, attention *reads* and feed-forward outputs are skipped only where their results cannot affect the next state or the stored keys/values. The state, boundary, previous-token and key/value updates are still performed. A synthetic causal test and an independent real-book 4K-token comparison confirmed **bitwise equality** of the state and future logits with the original full warmup.

| Book/split | Query token (actual) | R27 BPB | R28 BPB | R28 gain |
|---|---:|---:|---:|---:|
| 1444/dev | 33,137 | 2.554800093 | **2.553293768** | +0.001506325 |
| 1444/dev | 65,231 | 2.414187159 | **2.412804711** | +0.001382449 |
| 1444/dev | 132,169 | 3.010388142 | **3.009415548** | +0.000972594 |
| 710/historical test | 33,128 | 2.282822980 | **2.281614267** | +0.001208713 |
| 710/historical test | 65,000 | 2.520380786 | **2.518915177** | +0.001465608 |
| 710/historical test | 132,127 | 2.135792033 | **2.134911654** | +0.000880379 |

**Aggregate over precisely these six small windows:** 3,246 bytes, 1,772 predicted tokens, R27 `2.487997148` BPB vs R28 `2.486754448` BPB; weighted difference **−0.001242700 BPB**. All six windows and their associated non-space NLL values improved with R28. This demonstrates that the tiny R28 residual advantage can survive long causal replay on these particular historically seen texts, but does **not** establish whole-book transfer, semantic retrieval, statistical significance or author-disjoint generalization. Each book contributes three correlated windows.

## Exact subword overlap screening

A three-train-book vs six development/test-book scan excluded the first/last 512 subword tokens (boilerplate control). Roughly **1.5%–2.6% of 8-subword windows** in the six held books were also seen in the three training books. Of the 16-subword windows, at most two matches per held book appeared in the training sample (the other four had none). This is a useful exact-copy check but not proof of **author disjointness**, absence of paraphrases, or lack of prior PAL exposure.

## Key decisions and next engineering target

1. **Do not promote** the 2/4/8-horizon arms or the non-space-weighted variants. Long BPTT alone does not fix the objective and sample-budget problem.
2. **Preserve R27 and R28 recovery assets** and the complete inherited optimizer history; ship only source, verified logs, results and the existing compact R28 adapter in this round's cumulative ZIP (no duplicate 124MB checkpoint).
3. For a subsequent candidate, use precommitted and longer mixed-book training, better gradient isolation from whitespace, and a causally conditioned residual trust/gating objective; evaluate a sealed, author-disjoint corpus and document-level QA/semantic distractors before any frontier-language-model claim.

Reproduce experiments from the extracted R19 ancestor + R27 ZIP + this current cumulative overlay using `source/r29_horizon_match.py`, `source/r29_long_replay.py`, `source/r29_corpus_overlap.py`, and `source/test_r29_fast_replay.py`. The repository `master` branch carries the matching source and GitHub Actions causal checks.
