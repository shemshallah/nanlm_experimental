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
