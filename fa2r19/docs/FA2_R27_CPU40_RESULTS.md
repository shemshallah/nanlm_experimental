# FA2 R27 — distance-aware long-memory trust: verified CPU40 experiment

**Date:** 2026-11-12 US Pacific. **Status:** experimental R27 unified candidate, NOT a frontier-quality LM and NOT a promotion of the original accepted R18 ancestor.

## One continuing model

R27 continues the single FA2 PAL lineage. The full R27 checkpoint includes R18–R26 neural weights and history and the additional R27 causal counterfactual trust module. The 179 inherited R26 model tensors, 137 inherited optimizer states, and all 12 document-stream snapshots are preserved **bit for bit**. The complete checkpoint contains 12 optimizer groups and 144 Adam states. No separate R24/R25/R26 candidate binary is needed for recovery when paired with the original R19 full ancestor release.

The R27 trust module compares the inherited full R26 high-order span predictor with a near-only R26 counterfactual, both using the same R26 weights. Its new gate starts at identity. Following a rejected pilot that regressed on 33K, the corrected gate only changes predictions at query positions >= 131072; all earlier predictions are identical to R26 by construction. The gate is based on causal prefix features, never labels.

## Real-book CPU40 development selection

Two historical NaNROUTER training books (IDs **240,784**), three R27 selection development books (**866,970,1444**), 40 new R27-only optimizer updates. Same inherited R26 checkpoint, no backbone or R26 Adam updates. Development positions at 4096,8192,33000,65000,132000 (where available). The selected R27 was **update 40**, chosen on development BPB subject to near and 132K safeguards and correct-vs-donor ranking.

| Matched development | R26 BPB (lower) | R27 selected BPB | Delta |
|---|---:|---:|---:|
| Total (1168 bytes) | 2.393953530 | 2.393750619 | -0.000202911 |
| Near positions | 2.470752855 | 2.470752855 | exactly unchanged |
| 33K positions | 2.304551598 | 2.304551598 | exactly unchanged |
| 132K window (113 bytes) | 2.256431471 | 2.254334123 | -0.002097348 |

The first unrestricted 40-update pilot was **rejected**: its dev total 2.394423434 BPB exceeded inherited R26's 2.393953530, including a harmful 33K difference. That experiment was not the source of the released adapter.

## Reserved long-book probes — no further fitting

With selected R27 step 40 locked, the completed 132K probes used three different historically seen development books (IDs **8571,13505,15985**), not used in this R27 round's model selection. Their complete preceding prefixes were causally replayed. All three improved in these small windows:

| Book | Bytes | R26 BPB | R27 BPB |
|---|---:|---:|---:|
| 8571 | 113 | 1.952379294 | 1.951195420 |
| 13505 | 119 | 1.837473510 | 1.834854980 |
| 15985 | 119 | 2.591323543 | 2.577043445 |
| **Combined** | **351** | **2.130044757** | **2.123934459** |

Combined delta **-0.006110298 BPB**; correct-book vs wrong-book donor advantage remained positive at **+0.600969825 nats/token**. These tiny windows do not establish whole-book transfer or statistical significance. All books are from previously consulted Gutenberg/PAL material: **not a sealed, independent-author test**.

## Integrity and cold-restore audit

* Original immutable accepted R18 ancestor SHA-256: `dd040106568786e7fbad2a880f4010635b626b51c3697e2ab08712905a2de91f`.
* Immediate R26 full candidate SHA-256: `8cb6b5524fafcbdc6427e175000d490451a800287964ee509bf2b44aaea376fc` (R26 step90, R25 step120).
* R27 unified checkpoint SHA-256: `0f4cdb8ce41b62472b6c8408296b496dfb3e801b388289453c0cdcc720f5b3f8` (R27 step40).
* R27 recovery ZIP: `FA2_R27_UNIFIED_LATEST_RECOVERY.zip`, **80,268,534 bytes**, **97 entries**, SHA-256 `d948fa1cc0a09e07dfad7f2798189e2d41728b9aec9456390c444e0612a6da97`.
* Independent clean-directory cold restore: passed using original PAL R19 full ZIP + one R27 ZIP only; no intermediate R26 candidate binary present. Manifest entries, source restore, and optimizer-group audit passed.

## Reproduce

From a clean directory, extract **the original PAL R19 accepted full-release ZIP**, then overlay **the single R27 latest recovery ZIP**. The archive contains the original executed R27 pilot, frozen reserve probe, checkpoint constructor, cold-restore code, packages/manifests, and logs. With NumPy and CPU PyTorch installed:

```bash
PYTHONPATH=source python source/r27_restore_unified.py --root .
PYTHONPATH=source python source/test_r26_causal.py
# Historical R27 pilot and reserve probe are source/r27_local_pilot.py
# and source/r27_reserved_probe.py; the real-book pilot replay is CPU-intensive.
```

The R27 synthetic trust tests run in GitHub Actions from `source/test_r27_trust.py`; that test is maintained in GitHub source and is not included in the compact R27 recovery ZIP.

Reports: `reports/r27_realbook_local_pilot.json`, `reports/r27_reserve_long_132k.json`, `reports/r27_inheritance_audit.json`. Logs and per-member SHA-256 manifest are in the recovery ZIP.

## Frontier gaps and promotion gate

This is **long-distance lexical trust calibration**, not semantic retrieval. The inherited recurrent truncated BPTT remains two consecutive 12-word chunks. The quality gains are far too small and selective to support any frontier-language-model claim. Priority: author-disjoint independent new books and leakage screening; paraphrase and hard-negative semantic retrieval; then capacity-preserving recurrent/attention widening with longer BPTT, matched compute, and frozen test gates. Keep R27 experimental pending those results.