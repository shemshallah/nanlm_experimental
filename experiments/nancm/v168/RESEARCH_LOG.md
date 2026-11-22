# v168 research log (2026-11-22)

- Recovered v167 from cumulative ZIP and checked canonical inherited model fingerprint. Left `ACTIVE_MODEL.json` and v167 default entrypoint unchanged.
- Reused 300 raw TRAIN-book text windows of 384 bytes each; deterministic 80 train / 20 validation book split. Continued 250 AdamW steps with state preservation, candidate has better NLL but no top-1 lift and weaker top-5. Rejected for active use.
- Trained a causal scalar gate using TRAIN windows; native independent-book DEV top-1 fell to 40/480. Rejected.
- Probed signed oriented concept geometry and learned 32D lexical state in 16D+32D retrieval. Signed-only 22/480, mixed signed+lexical 41/480 standalone top-1; 192/480 parent+ANN top32 union on 480 DEV positions. These are retrieval/support diagnostics, not native decoder scores.
- Built fingerprinted 51.8MB TRAIN-only signed ANN index from 199,956 context keys. Integrated `wordfield168_ann_api.cpp` into native 27-way inference; exact strength=0 v167 rollback tested.
- DEV β=.05 native: 480 pos top1 46→46, top5 94→93, top32 150→158; 1,394 pos top1 141→141, top5 269→267, top32 465→477. Sampled reused TEST top1 46→47, top5 104→103, top32 162→165. Symbol loss improves modestly, but top1 does not on DEV; not promoted.
- Native regression verified 450 exact v167 optout vectors, 1440 normalized prefix distributions, 1440 future suffix parity, 4545 historical document fallback prefixes, and 5 fail-closed configurations.
- Book-level novelty remains unproven. Further rounds should target first-choice learning from larger raw independent-book corpora with calibration separate from evaluation and book-disjoint provenance.