# v167 lineage and data provenance

- Direct predecessor: promoted NaNCM v166 `v166_lexical_birth.pt` and `.bin`, SHA-256 unchanged. Model trained from 500-book TRAIN-only *lossy conditional index*, with plus/minus/lower/upper originally inherited from v165 including AdamW moments. v164 conditional memory and all v148/v158/v162 inherited native checkpoints are unchanged.
- v167 active construction: build-only native inference gate; **no checkpoint weights are trained, merged, or overwritten**. Trust score uses only pre-target Q166 and v164 proposal words, and past-book domain statistics.
- v167 hard-negative train candidate: same v166 parent, extra 350 AdamW steps on TRAIN-index context groups, 768 held-out TRAIN-index diagnostic contexts (not book-disjoint). Saved separately; not active because DEV top-5 regressed.
- Full native tests: v166 exact rollback on 951 prefix comparisons; documentation fixture exactly matches on 4,545 prefixes despite relaxed domain1.4. Invalid configurations fail closed.
- DEV: 24 distinct books ×20 deterministic positions, previously inspected in v164/v166; selection uses DEV, NOT TEST.
- TEST: 24 separate books ×20 positions, previously inspected in older versions. Frozen v167 configuration recorded before this TEST call in `FROZEN_DEV_DECISION.json`; no tuning on TEST in v167.
- Limits: exact duplicates checked in older v160 pipeline, but near-duplicates and overlap with inherited neural training are still unresolved. No claim of pristine certification or full 256-byte BPB.