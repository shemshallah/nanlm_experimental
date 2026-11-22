# NaNCM — active development baseline

**Promoted by explicit user instruction on 2026-11-21:** NaNCM v158 Causal Whole-Word Cloud Pointer is the active **development** baseline. This is not a claim of independent-book certified production.

- Preserve the earlier v148 fallback and all inherited 128+128 signed cloud weights / AdaGrad and AdamW optimizer histories.
- Native v158 historical documentation DEV strict word top-1 21.5924%, top-5 33.7382%; reused TEST top-1 43.4109%, top-5 52.8424%. These historical fixtures were repeatedly used during development and are not independent-book certification.
- v158 active setting: V158_POINTER_THRESHOLD=.8, V158_POINTER_BUDGET=.35, V158_TOPIC_BUDGET=.015.
- v159 experimental extension: second byte-context worldline pointer and document-local spelling adaptation. Default V159_BYTE_BUDGET=0, V159_CHAR_BUDGET=0, providing exact v158 rollback.
- v159 byte-pointer .03/minmatch 12: historical DEV top-1 unchanged, top-5 33.7382% -> 33.8731%; reused TEST top-1/top-5 unchanged, top-32 64.5995% -> 64.7287%. Unpromoted.
- Rejected: aggressive v159 .12 byte-pointer (reused TEST subset top-1 regression) and unrestricted .12 local spelling (historical DEV top-1 regression).
- Required before independent production promotion: verified NaNROUTER book-disjoint DEV/TEST, 256-byte BPB, BPE, strict whole-word gates, rollback and Android native integration.

The complete recoverable v158 and experimental v159 ZIP artifacts are kept in the ChatGPT Library. This status file is not a substitute for those optimizer/checkpoint archives. Maintain experiments in folders on **master**; do not create competing checkpoint lineages.

## v160–v162 independent-book research — 2026-11-21

The **active development baseline remains v158**. v162 is **opt-in research only**, not independently certified production. v148 remains the protected fallback. Full trained model and optimizer checkpoint ZIP is held in the ChatGPT Library.

- Verified NaNROUTER short-story batch 0004 SHA-256 `d6ce3a23757e85fe1b9487ef984a5ba81db9c2dc6e962ac7c5df2960480e0054`; split by deterministic book filename hash: 500 TRAIN / 51 DEV / 64 TEST. Exact duplicate check passed; near-duplicates and full inherited-training overlap are **not fully screened**.
- GitHub Actions book-only route and index training passed: [run 44980376171](https://github.com/shemshallah/nanlm_experimental/actions/runs/44980376171) and [run 44980989567](https://github.com/shemshallah/nanlm_experimental/actions/runs/44980989567). Source: `experiments/nancm/v160_book_gate.py`; workflow: `.github/workflows/nancm_v160_books.yml`.
- Long-context v160 route fires on ~1.23% of unseen-book words with ~76–78% precision *conditional on firing*, not model-wide accuracy. v161 1/2-word backoff route fires on 58.64% of DEV words with 19.54% standalone precision.
- Paired native 96-beam scoring on 480 word positions from 24 DEV books and 480 from 24 TEST books: v158 DEV top-1 30/480 (6.25%), TEST 28/480 (5.83%); opt-in v162 causal domain-arbitrated fusion DEV 39/480 (8.13%), TEST 40/480 (8.33%). Historical documentation DEV top-1 160/741 and reused TEST 336/774 preserved exactly: 4,545 prefix probability checks were byte-for-byte identical to the v158 default for these contexts. Domain scoring uses observed history, never target text.
- v162 native C++ regression: 360 exact strength-0 rollback checks, 720 enabled heldout-book normalized prefix checks, clean extraction/recompile, original model/optimizer checkpoint fingerprint parity. Full byte BPB, token/BPE metrics, near-duplicate screening, and larger book-disjoint neural finetuning remain **open**.
- Research archive: `NaNCM_v162_CAUSAL_DOMAIN_ARBITRATED_CLOUD_RESEARCH_20261121.zip`, SHA-256 `9ddb8f27b27cdc942f891364a1c16ab5444cfb09225bb53ebe703eb12a22fcc5` in ChatGPT Library. Run `make test-v162` from its extracted source. No production promotion claimed.
