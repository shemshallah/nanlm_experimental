# FA1 R7.22 — ONE ACTIVE R7 ARCHITECTURE ON `master`

Canonical sources: [R7_CURRENT.md](R7_CURRENT.md), [unified model](source/fa1_r722_unified.py), [associative reader](source/fa1_r722_semantic.py), and [report](FA1_R7p22_UNIFIED_1M_ASSOCIATIVE.md). R7.22 extends R7.21 in place: 105 tensors, five original optimizer histories, and exactly-off optional 128K/1M embedding-associative routes. The accepted R7.13 checkpoint remains at step 14,416; 48 gate-training updates were rejected, and independent-book validation is pending. Prior R7 files are historical reference implementations, **not competing models**. No new Git branch or model promotion.

---

# FA1 — formerly PAL, R7.12 baseline and R7.13 hierarchical experiment

This subtree is the **FA1** research lineage. `source/` and `reports/` mirror the full textual PAL R7.12 release, plus the current hierarchical FA1 experiment. No NaNLMind model is merged into FA1.

## Source of truth / checkpoint policy

The accepted full PAL checkpoint remains `checkpoint/pal_active.pt` at **14,416** joint updates (SHA256 `54e835b00642013d4f1bf486a8150eb1ff06c547565771bd21933e143d050f2d`). All **five inherited optimizer histories** remain unchanged. The R7.11 writer adapter and R7.12 LM calibration adapter are experimental, not promoted. FA1 hierarchy adds a small auxiliary router adapter; it does not replace the accepted weights.

Original **PAL R7.12** ZIP: 240,360,600 bytes, 246 entries. GitHub's connected file-writing interface currently supports UTF-8 text commits, not attaching this binary ZIP or 94.7 MB checkpoint as a release asset. This GitHub subtree therefore contains **source/reports/provenance only**, *not* a runnable independent full-checkpoint release. See `legacy_R7p12/ARCHIVE_INVENTORY.json` for checksums and obtain the full R7.12 ZIP from the existing ChatGPT Library artifact.

## Hierarchical CPU experiment

```
# Unpack the full R7.12 release into ./fa1-working and place this source tree there.
cd fa1-working
PYTHONPATH=source OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python source/fa1_hierarchical.py --root . --train 512 --dev 512 --test 512 \
    --updates 192 --batch 16 --threads 2
PYTHONPATH=source OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python source/test_fa1_hierarchy.py --root . --count 128
PYTHONPATH=source OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python source/fa1_hierarchical_lm.py --root . --dev 512 --test 512
```

Expected: 8 × 768-token historical segments, query-driven top-2 readout with the same trained rank-64 directed writer. **Historical lexical recall** improves vs fixed last-768. It does not prove semantic retrieval, and the joint LM readout still **fails** the correct-vs-unrelated-memory gate.

| Historical 512-query retrieval | Development | Test |
|---|---:|---:|
| Fixed last 768 CE | 3.8381 | 4.0098 |
| FA1 top-2 routed CE | 3.2642 | 3.4273 |
| FA1 correct-memory top-1 | 21.09% | 18.95% |
| Unrelated-memory CE | 3.6084 | 3.7743 |
| Reversed-order CE | 4.5903 | 4.4475 |

Adversarial temporal-conflict query: top-1 segment selection **4.35%** on 115 evaluated events even though top-2 includes it; embedding-neighbor perturbations: top-2 source retrieval **62.5%**. These failures guide the next work: revision-aware routing and true paraphrase tests on document-disjoint NaNROUTER books.

## GitHub and release discipline

Every future FA1 training round should update code, evaluation reports, provenance, and the pointer to the **one accepted checkpoint**. Do not publish unverified candidate overlays as accepted weights. Check memory-specific LM advantage and held-out book splits before promotion. GitHub updates occur during active chats, not as an autonomous background process.

## R7.14 fast CPU diagnostic (2026-11-12)

Two CPU candidate adapters were tested on the unchanged step-14,416 checkpoint and failed the correct-vs-unrelated-memory LM promotion gate. The accepted FA1 weights and all five optimizer histories remain unchanged; FA2 is separate. See [FA1 R7.14 CPU results](FA1_R7p14_FAST_LOCAL_CPU.md). New scripts/reports are distributed in the conversation's 15.6 KB minimal source ZIP; they are not yet present in this GitHub subtree.

## R7.15 query-conditioned mirror CPU experiment (2026-11-12)

**Not promoted.** The original step **14,416** full checkpoint and five optimizer histories are unchanged (SHA256 `54e835b00642013d4f1bf486a8150eb1ff06c547565771bd21933e143d050f2d`). R7.15 adds a candidate query-conditioned rank-64 S+/S- reader/writer trained from LM loss using a temporary extra group in the fifth optimizer; it does **not** jointly update the original trunk.

- **Historical natural-language LM:** 128 CPU updates; no development NLL plus wrong-history-control gate passed. Baseline dev NLL 4.618318, unsuccessful step-128 candidate 4.845736. Reject candidate, zero out its gate.
- **Historical contextual cache:** 256 development samples; all 20 tested trust settings lost to original LM and failed memory controls.
- **Synthetic long-range key/value copy:** best step 112/128; on held-out random keys, NLL 1.0782 versus an explicitly **uniform** baseline 10.3735. Mirror-rail ablation NLL 1.1674. **This is not real-language LM gain.**
- **Synthetic-to-natural transfer:** historic dev NLL rises from 5.0689 to 5.1561 despite correct-memory beating unrelated-memory by 0.0207. No promotion.
- **Integrity:** five new structure and target-leakage tests pass. No independent new-book evaluation was performed; NaNROUTER corpus documents are GitHub **Release assets**, not committed directly under `corpora/`.

See [full R7.15 CPU report](FA1_R7p15_CPU_REPORT.md), `source/fa1_r715_*.py`, `source/test_fa1_r715.py` and `reports/FA1_R7p15_*.json` for fully reproducible small CPU pilot. The accepted ~95-MB checkpoint is not in GitHub. The small cumulative R7.14/R7.15 source/reports ZIP resides in the conversation/Library, to use alongside the full accepted R7.13 archive. **FA2.2 remains separate.**

## FA1 R7.16 — causal 32K trust router (2026-11-12)

**Candidate only; accepted weights remain step 14,416.** The original checkpoint SHA-256 is `54e835b00642013d4f1bf486a8150eb1ff06c547565771bd21933e143d050f2d`, with five optimizer histories preserved. A separate small, dev-selected router trained against LM NLL gave 5.092974 versus 5.162442 development NLL and 5.655632 versus 5.789165 historical test NLL (both **nats/word**); correct auxiliary history beat unrelated memory by 0.077321 nats/word on historical test. The earlier 8K-only route failed to improve development NLL; 32K-only explained the gain. Structural and integrity checks: 9/9. All legacy splits have been repeatedly inspected, so **no new independent-book claim or frontier-model claim** is justified. This adapter is not a replacement accepted model.

Source: [R7.16 report](FA1_R7p16_FRONTIER_CPU.md), [model adapter (base64 text)](source/fa1_r716_candidate.pt.b64), [training program](source/fa1_r716_router.py), [new-book book-cluster evaluator](source/fa1_r716_newbooks.py), [tests](source/test_fa1_r716.py), and [JSON metrics](reports/FA1_R7p16_225.json). The manual [whole-book audit workflow](../../.github/workflows/fa1-r716-frontier.yml) is fail-closed until the verified 214.5MB accepted FA1 R7.13 checkpoint archive is separately available as a GitHub Release tagged `fa1-accepted-r7.13`; GitHub currently stores only code, reports, and the small unpromoted adapter. NaNROUTER books are download-only Release assets. FA2.2 is separate.

## FA1 R7.17–R7.18 — future-safe 128K readout

**Accepted checkpoint remains at 14,416 joint updates.** The original full binary SHA256 is `54e835b00642013d4f1bf486a8150eb1ff06c547565771bd21933e143d050f2d`, with all five inherited optimizer histories unchanged. R7.16's small dev-selected adapter is frozen and unpromoted; R7.18 is **a new additional experimental overlay**, not a second independent LM or full-checkpoint replacement.

**R7.17 was rejected and its wrong-history control diagnosed as invalid at 128K.** On the repeatedly used 256-example historical dev sample, 33.2% of donor windows were future positions whose far-memory span could include the target. The earlier negative correct-vs-wrong result is not interpretable as a clean causal comparison. Reproduction source and JSON reports remain under `source/fa1_r717_router.py` and `reports/FA1_R7p17_*.json`; **do not use that rolled-donor diagnostic as a valid proof of 128K failure**.

**R7.18 uses disjoint 32K–128K older history with future-safe controls**: word-order shuffling inside the existing historical window, and donor histories from the separate legacy training split. Its 1,024/512/512 historical train/dev/test experiment ran 240 CPU updates, selected step 48 by dev gate, and achieved:

| NLL nats per word ↓ | Accepted FA1 | Frozen R7.16 near | R7.18 near + 128K far |
|---|---:|---:|---:|
| Historical development | 5.235429 | 5.146455 | 5.105531 |
| Historical test | 5.945565 | 5.757913 | **5.516152** |

Test correct-history advantage over shuffled history: +0.290978 nats/word; over distinct legacy training-split donor: +0.264486. The additional old-history gain beyond the frozen R7.16 reader is 0.241761 nats/word (4.20% relative); vs accepted parent, 7.22% relative. **These historical splits have been repeatedly consulted; this is not independently validated generalization.** Twelve structural/causal tests pass.

Full report: [R7.18 corrected controls and caveats](FA1_R7p18_128K_CORRECTED_MEMORY.md). Source: `source/fa1_r718_far_residual.py`, `source/fa1_r718_newbooks.py`, `source/test_fa1_r718.py`; immutable candidate binary is recoverable from `source/fa1_r718_full_candidate.pt.b64` after verifying its SHA256 `f4d9175ff202d9ec1ff3a06dcddfd3d262cea25f8441fa5dab972a313309aa9f`. JSON metrics under `reports/FA1_R7p18_*.json`. A manually dispatched [sealed independent-books workflow](../../.github/workflows/fa1-r718-newbooks.yml) can evaluate new NaNROUTER books but **currently cannot run** until a verified full accepted R7.13 checkpoint archive is hosted where a GitHub runner can access it. It refuses to promote anything automatically. FA2/FA2.2 and NaNLMind remain untouched.


## FA1 R7.19 — indexed 128K CPU continuation (unpromoted)

R7.19 adds a causal incremental weighted 1–4-gram index over the 32,769–131,072-token far band, preserving the accepted R7.13 step-14,416 checkpoint and frozen R7.16/R7.18 experimental adapters. On synthetic CPU retrieval, the maintained index was 45.1x faster **after** its 1.2–2.5s construction cost; this is not end-to-end LM speed. Local evidence: 11/11 new tests, 12/12 R7.18 regression tests, and equal gate/NLL on four actual historical-checkpoint positions. Independent NaNROUTER books remain untested; no promotion. See [R7.19 report](FA1_R7p19_INCREMENTAL_128K_MEMORY.md), [source](source/fa1_r719_streaming_index.py), [tests](source/test_fa1_r719.py), and [CI](../../.github/workflows/fa1-r719-cpu-index.yml).


## FA1 R7.20 — packed causal 128K CPU memory (not promoted)

R7.20 adds a collision-free packed uint64 1–4-gram inverted index with ~4.7 MB of sorted NumPy index arrays. The [R7.20 report](FA1_R7p20_PACKED_128K_CPU.md), [runtime](source/fa1_r720_runtime.py), [packed index](source/fa1_r720_packed_index.py), and [CPU tests](source/test_fa1_r720.py) are published. On two 1,024-position historical checkpoint benchmarks in opposite execution orders, end-to-end CPU acceleration was 1.16–1.17x, with max final NLL discrepancy 9.54e-7 nats/word. The source has an `auto` fallback for short and sparse queries. Accepted FA1 remains at step 14,416 with five optimizer histories unchanged; independent book evaluation is still blocked pending a full R7.13 checkpoint GitHub Release.


## R7.21 ONE UNIFIED R7 ARCHITECTURE

**Active R7 source of truth:** [`R7_CURRENT.md`](R7_CURRENT.md), [`FA1_R7p21_UNIFIED_ARCHITECTURE.md`](FA1_R7p21_UNIFIED_ARCHITECTURE.md), and [`source/r7_current.py`](source/r7_current.py). One model, one R7.21 recoverable candidate, five inherited optimizer histories, on `master` only; old R7.14–R7.20 files are reference implementations, not alternative models. The accepted step-14416 model is unchanged; unpromoted R7.21 candidate achieved exact 1,024-position historical parity. Joint 96-step CPU pilot was rejected on development. Whole-book NaNROUTER validation remains mandatory for promotion.
