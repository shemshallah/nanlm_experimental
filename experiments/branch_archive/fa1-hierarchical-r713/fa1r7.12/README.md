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
