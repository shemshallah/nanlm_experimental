# FA2 R27 — Unified long-range trust checkpoint (step 40)

2026-11-12 Pacific. **Experimental candidate, not a frontier-model promotion.**

This release preserves the SAME PAL network lineage R18 → R21 → R22 → R23 → R24 → R25 → R26 → R27. It does not contain a second language model. The R27 gate is an additional parameter group on the existing optimizer, not an optimizer restart.

## Exact inheritance

- Original immutable R18 (in the separate PAL R19 ancestor) SHA-256: `dd040106568786e7fbad2a880f4010635b626b51c3697e2ab08712905a2de91f`.
- R26 immediate parent SHA-256: `8cb6b5524fafcbdc6427e175000d490451a800287964ee509bf2b44aaea376fc`.
- R27 complete unified checkpoint SHA-256: `0f4cdb8ce41b62472b6c8408296b496dfb3e801b388289453c0cdcc720f5b3f8`.
- Exactly preserved **179 R26 model tensors**, **137 R26 AdamW optimizer states**, **12 document-stream snapshots**.
- One unified model; one optimizer with **12 parameter groups / 144 Adam states** after R27's seven new Adam entries.
- Original R25 and R26 counters remain 120 and 90; R27 selected at 40/40 pilot updates.

## Results (exploratory, small samples)

The frozen R26 model and trained R27 use identical prediction windows and book tokens:

| Cohort | R26 BPB ↓ | R27 BPB ↓ | Scope |
|---|---:|---:|---|
| Selection development | 2.393953530 | **2.393750619** | 3 historical development books, 1,168 bytes total |
| 132K selection window | 2.256431471 | **2.254334123** | 113 bytes |
| 132K reserved long books | 2.130044757 | **2.123934459** | 3 books, 351 bytes |

The reserved books are Gutenberg #8571, #13505, #15985, excluded from this R27 selection. All three improved. These authors were **not** globally sealed from earlier PAL experiments; the result is not independent blind replication. This model does not demonstrate semantic retrieval, long-gradient propagation, or frontier language understanding.

A rejected preliminary gate without a 128K activation boundary caused an overall development regression (R26 2.393953530 versus R27 2.394423434). R27's current gate is *causally inactive for all query positions below 131072*, preserving R26's earlier gains exactly. This guard was selected after observing a development failure; do not generalize it as a proven best threshold.

## Cold restore

1. Extract the **original** `PAL_R19_SEMANTIC_FRONTIER_AUDIT_ONE_CHECKPOINT_FULL` release into an empty directory. It supplies the immutable accepted R18 ancestor, book corpus, vocabulary, and initializer assets.
2. Overlay **this R27 recovery ZIP** into the same directory. **No R24, R25 or R26 binary is needed**: the one R27 full checkpoint contains all continuing model weights and optimizer moments.
3. In the combined directory execute:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=source python source/r27_restore_unified.py --root .
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=source python source/test_r26_causal.py
```

You can reproduce gate selection from the local source `source/r27_local_pilot.py`; it requires the original R26 parent and will not modify that parent. `source/r27_reserved_probe.py` reruns the three reserve-book windows from the selected small adapter and original R26. The feature cache `reports/r27_extracted_cpu_cache.pth` holds **real extracted R26 predictions**, not synthetic score data; it is included to facilitate faster auditing. The released complete R27 checkpoint is the source of truth after cold restore.

Only one latest R27 archive and the original distinct R19 ancestor are required for recovery. All archive members are listed in the manifest and SHA256SUMS. GitHub stores the source, tests and results, not the 129-MB checkpoint binary.