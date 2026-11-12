# FA1 R7.16 — 32K causal memory, LM-aligned trust routing, and independent-book gate

Date: 2026-11-12  |  Track: **FA1 only** (not FA2/FA2.2 or NaNLMind)

## Result and checkpoint decision

**R7.16 experimental adapter passes both historical development gates, but is NOT promoted to the accepted unified model.** The full accepted model remains at **14,416 joint updates**, SHA256 `54e835b00642013d4f1bf486a8150eb1ff06c547565771bd21933e143d050f2d` (five original optimizer histories). The historical splits have already been repeatedly consulted; a new NaNROUTER whole-book evaluation is required before promotion.

The single candidate adapter was trained from a frozen copy of the accepted trunk, attached as an additional parameter group of the **existing fifth optimizer**. All pre-existing optimizer tensor values and group settings were checked to be unchanged. R7.15's rejected candidate was not promoted; R7.16 reuses the accepted original model, never a new initialized full model.

## Architecture upgrade actually implemented

* Existing inherited model: 96D learned word features, 32K-word hierarchical predictor, rank-64 mirror, 8,192-token historical horizon including a 2,048-token recent window.
* R7.16 readout: **8,192 or 32,768 historical tokens** (including recent window), with sparse conditional suffix matching over 1-, 2-, 3- and 4-word historical keys. At the 32K scale, 30,720 causal older tokens are eligible for the auxiliary reader; the recent 2,048-token region is excluded to avoid double-counting/local target leakage.
* Each of 8 routes stores *target-free* match count, recency, mode concentration, diversity entropy, and the inherited trunk's probability of the **history-only modal next word**. Each proposed value distribution is formed entirely from earlier causal keys and subsequent observed historical values. No target participates in indexing, selecting a mode, or computing the gate.
* A 48-hidden-unit gate learns a distribution over no memory + 8 memory paths, capped at 35% total interpolation. It learns by exact next-word mixture NLL with a small correct-versus-unrelated-history contrast. New weights and Adam moments are in a ~49KB candidate overlay, not a second language model.
* This is a **sparse 32K conditional-memory capability**, **not** 32K dense transformer attention, 32K-token subword modeling, or frontier reasoning.

## Finished matched historical CPU evaluation

One 225-update CPU run, with 1,024 training positions, 512 development positions, and 512 historical test positions. Only development selected the 75-update checkpoint; test metrics were computed after selection. The old accepted checkpoint was hash-checked before and after the full run.

| Metric (nats per **word**, lower is better) | Original accepted FA1 | Selected R7.16 | Gain |
|---|---:|---:|---:|
| Historical development NLL | 5.162442 | 5.092974 | 0.069468 (1.35%) |
| Historical test NLL | 5.789165 | 5.655632 | 0.133533 (2.31%) |
| Historical test, unrelated auxiliary memory | 5.789165 | 5.732954 | 0.056212 (0.97%) |

Correct-vs-unrelated auxiliary history NLL advantage is **0.030900** on development and **0.077321** on historical test. The inherited base decoder kept its **correct original memory** in both arms; only the NEW auxiliary memory was switched to the unrelated donor. This is an *incremental-route* control, not an ablation of all original FA1 memory.

### Scale ablation on development (same selected gate)

| Memory route | NLL | Gain vs original | Control advantage |
|---|---:|---:|---:|
| Original (no R7.16 route) | 5.162442 | 0 | — |
| 8K route only | 5.163350 | **-0.000907** | -0.001212 |
| 32K route only | 5.092206 | **+0.070237** | +0.031635 |
| All 8K and 32K routes | 5.092974 | **+0.069468** | +0.030900 |

The longer history is doing the useful work in this small experiment; no evidence suggests that simply increasing local cache strength would have worked.

### Additional pilot and incompletely finalized run

An early 128-train/64-dev/64-historical-test, 64-update smoke experiment passed its gates (dev gain 0.031672, historical test gain 0.028746). A subsequent 1,000-update attempt showed its best sampled development gain 0.063312 but did **not** finish the final test/report within its execution window. Neither candidate was selected for this release. The completed 225-update run with independent development selection is the reproducible R7.16 result.

## Structural tests / failure controls

**9/9 passed**: (1) 32K older-event retrieval with 8K miss; (2) excluded recent-history events cannot enter the new memory path; (3) changing target tokens does not change indexing or gate-input features; (4) edits in the last 2,048 historical tokens do not affect the auxiliary memory; (5) no-evidence routes have exactly zero gate weight; (6) gate off reproduces inherited NLL; (7) both scale ablations are finite; (8) exact causal 1/2/3/4-gram indexing; (9) completed report and optimizer-preservation checks.

The accepted full binary SHA256 and all inherited optimizer tensor states were checked unchanged. A toy 120-document importer/auditor fixture passed whole-file content-SHA disjointness and document-boundary checks (108 train / 6 dev / 6 test). **The toy files are not NaNROUTER books and do not count as independent validation.**

## True held-out NaNROUTER book gate — prepared, NOT executed

The NaNROUTER public-domain texts are GitHub Release assets, such as `corpora-short-stories-0001`, not files committed directly in the repository. The local container cannot access those external binary assets. The existing importer has an independently recorded expected archive SHA256 and SHA-checked safe extraction. `fa1_r716_newbooks.py` then:

1. Recreates 90/5/5 whole-book hashed train/dev/test partitions from newly downloaded individual Gutenberg files, with explicit document-boundary masks.
2. Enforces unique exact SHA256 content across partitioned books and 16-word known-vocabulary fingerprint overlap screening against all inherited train/dev/test corpora. A clean fingerprint screen is **not** proof of no overlap, especially through unknown-token regions.
3. Applies the frozen *historical-dev-selected* adapter only; computes correct and independently rerouted unrelated-memory NLL and equal-book weighted bootstrap confidence intervals.
4. Allows **review**, not automatic promotion, only if word-level gain, memory advantage, book-cluster confidence bounds, and overlap screens all pass.

The GitHub Actions workflow `fa1-r716-frontier.yml` is manual and fail-closed. **It cannot run to completion until the verified 214.5MB R7.13 full archive is available on a GitHub Release** tagged `fa1-accepted-r7.13`; GitHub currently holds source/reports and the small candidate overlay text, not the accepted binary. Do not mistake a successful import-fixture audit for a real NaNROUTER evaluation.

## Scaling ladder from a measured capability to frontier-relevant research

1. **Seal the data:** import 500+ distinct NaNROUTER books, hash/deduplicate, and pass book-cluster gain and adversarial history tests. Evaluate non-space accuracy and real raw-byte BPB separately; current numbers are word NLL, **not** BPB.
2. **Make 32K/128K retrieval streaming:** replace per-query rescans of up to 30,720 tokens with an incremental, causally aged sparse 1-4-gram/semantic index and exact parity tests; then test 32K, 64K, 128K contexts and retrieval at matched age distributions.
3. **Integrate the successful readout into one accepted checkpoint** only after the independent book gate: store router state and its fifth-optimizer group with the original learned trunk and all five existing states in one atomic release, while retaining exact inherited model predictions when the gate is off. Do not reset training moments.
4. **Expand trainable capacity without erasing it:** zero-initialized residual modules, width migration 96→192→384, rank-64→128/256 S+/S− memory, 4-8 conditional experts. Require exact initial-logit identity, old-optimizer mapping checks, matched-parameter controls, and genuine held-out BPB improvement before promotion. These *capacity expansions are proposed*, not yet implemented.
5. **Train on diverse raw text, not synthetic recall alone:** NaNROUTER Gutenberg + domains/code/math, with a true byte/subword backoff head, 1M/5M/10M-update milestones subject to compute and validation gates. Compare scaling-law improvement per CPU/GPU hour and against matched dense/recurrent baselines.
6. **Capability transfer:** held-out paraphrases, hard semantic negatives, code QA, multi-hop source attribution, and 8K→128K extrapolation. No frontier claims until these transfer and real-world evaluations improve.

The result is a useful first evidence that *longer causal context can improve FA1 language loss* after several failed short-memory designs. It does not imply parity with frontier language models or fluency.

## Reproduction (CPU, 2 threads)

Unpack the accepted FA1 R7.13 full ZIP under one working directory and overlay this source tree into it. In the directory containing `checkpoint/` and `source/`:

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
python source/test_fa1_r716.py
python source/fa1_r716_router.py --root . --train 1024 --dev 512 --test 512 \
  --steps 225 --eval-every 25 --batch 8 --train-batch 96 --threads 2 \
  --output reports/FA1_R7p16_225.json --overlay adaptation/fa1_r716_225.pt
```

Parent ZIP SHA256: `99f25e5ff604492c6c043335eddbf638415a423f43d94f5c9a426f5a283a2728`.
Accepted parent `checkpoint/pal_active.pt` SHA256: `54e835b00642013d4f1bf486a8150eb1ff06c547565771bd21933e143d050f2d`.
Selected **candidate overlay** SHA256: `436ff87fae69ef1f44b53e2ffde92313da408342f65a1897c6eaff1876f9c09c`.

The reproducibility ZIP contains only current source, reports, one small unpromoted adapter, logs, SHA256 manifest, and instructions; the large accepted checkpoint stays in its verified original archive. No nested ZIPs, binary databases, or FA2 assets.