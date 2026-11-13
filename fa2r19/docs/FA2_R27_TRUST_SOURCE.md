# FA2 R27 — unified counterfactual long-memory trust (source integration)

Date: 2026-11-12. **Implementation candidate only; no new checkpoint or
real-book BPB result yet.** R26 step 90 remains the accepted experimental
starting artifact, not a frontier-performance claim.

## Architecture

R27 adds ONE trainable trust module to the existing FA2 PAL network.
It retains all R18–R26 weights, all 137 R26 Adam states, the 11 original
optimizer groups, and 12 saved document-stream states. The 12th optimizer
group holds only the new R27 gate.

The gate compares (a) R26 using all six exact phrase channels, including
its 131072-token range, against (b) the **same** R26 with its three
long-horizon channels masked. It can interpolate back toward the
near-only counterfactual when the far-history estimate is unreliable.
All features are already causal R26 evidence (counts, mode concentration,
age, base-model agreement). No target labels are passed into gate features.
At initialization R27 is exactly R26, including full-vocabulary logits.
The original R26 model remains unmodified by default if R27 is absent.

This solves no semantic problem by itself: R27 is specifically a reliability
experiment. It must be paired later with author-held-out semantic matching
and substantially longer recurrent gradient horizons for broader claims.

## Required gated CPU experiment

After overlaying the R26 release atop the original PAL R19 accepted release:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=source \
python source/r27_cpu_train.py --root . --steps 100 \
 --train-books 4 --dev-books 5 --anchor-book 1444 \
 --offsets 4096,8192,33000,65000,132000
```

The command extracts actual PAL-causal R26 predictions, rather than
fabricating scores. It rejects missing 132K development coverage by default.
The selection gate compares **R27 directly to R26**, requires a dev BPB
improvement, retains the correct-vs-frequency-matched-wrong-book advantage,
does not worsen near or 132K-band BPB beyond 0.00001, and bounds non-space
NLL regression. If none passes, the command saves a report but writes
**no R27 checkpoint**. It never modifies the R26 starting file.
Historical test books are not part of selection.

If selected, R27 writes a **single complete experimental unified checkpoint**
at `reports/fa2_r27_unified_candidate.pth` and a JSON report. Before that,
it verifies all inherited model tensors and optimizer moment fields exactly.

## Evidence and release policy

- Source and synthetic causality checks are not measured language gains.
- The R26 book 1444 132K regression remains an unresolved baseline
  until real-book R27 measurements show otherwise.
- Do not mark R27 promoted, archive R26, or claim semantic/frontier gains
  merely because this implementation compiles.
- A later genuine-new-author sealed set and negative controls are required.

## R27 exploratory CPU pilot (not selected)

The first 40-update real-book pilot used training books 240 and 784;
development books 866, 970, and 1444. It tested fixed near, 33K,
65K and 132K offsets when present. At update 40, dev-window R26 vs R27
was near 2.470753 vs 2.470754 BPB; 33K 2.304552 vs 2.306472 BPB;
132K 2.256431 vs 2.254587 BPB. Overall R26 2.393953 vs R27
2.394423 BPB. No R27 candidate was selected. The pilot's final
archive audit crashed on a Python tensor-equality assertion after the
measurements; those logs therefore do not constitute a complete release.

The revision above **permits R27 only from query position 131072 onward**,
leaving earlier R26 results untouched. This guard is motivated by
observed development failures, not an independently confirmed transfer
gain. Fresh-author validation is still pending.
