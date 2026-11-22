# v168 claim ledger

| Claim | Evidence | Status |
|---|---|---|
| 199,956 TRAIN-only context keys and 733,069 candidate rows indexed | `results/v168_signed_ann_build.json` | Verified locally |
| Learned signed cloud participates in concept key, with 16D S+/S− projection | `ann_signed168.py`, `build_signed_ann168.py`, `wordfield168_ann_api.cpp` | Implemented |
| Native exact opt-out / normalized mixture / no future leakage | `results/v168_ann_native_regression.json` | Passed |
| DEV 480 top-32 150→158, top-1 46→46 at β=.05 | `results/v168_native_ann_dev_{0_0,0_05}.json` | Measured, small reused DEV sample |
| DEV 1,394 top-32 465→477, top-1 141→141 | `results/v168_native_ann_dev_all_{0_0,0_05}.json` | Measured, same 24 DEV books |
| Reused TEST top-32 162→165, top-1 46→47 | `results/v168_native_ann_test_{0_0,0_05}.json` | Measured, not pristine |
| v168 improves true whole-model full-byte BPB | Not evaluated | OPEN |
| v168 provides reliable generalization or ≥80% strict top-1 | Not established | OPEN |
| v168 merits promotion | DEV top-5 regression and no top-1 gain | NO |