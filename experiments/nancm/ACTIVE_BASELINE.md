# NaNCM — active development baseline

**2026-11-22 USER-DIRECTED DEVELOPMENT PROMOTION: v166 is now the active development baseline at V166_LEXICAL_BIRTH=0.05.** This is *not* production certification and does not assert a strict top-1 accuracy breakthrough. The previously active v158 checkpoint remains protected as rollback; v148 remains the legacy fallback. v164 remains the pretrained parent inside v166; the trained v165 signed concept tensors and optimizer moments are inherited by v166 and must not be reset.

- Native shared library: `bin/libwordfield166.so`, unchanged `wf148_*` ABI. Configure parent model, geometry, signed cloud, v157 word/spelling heads, v162/v164 causal domain and distribution, then set `V166_LEXICAL_BIRTH=.05` and `V166_BIRTH_MODEL=checkpoints/v166_lexical_birth.bin`.
- This conservative profile preserved DEV strict top-1 45/480 vs v164 45/480, improved top-5 84→86, and reduced spelling/EOS bits/symbol 2.880214→2.624520. Reused TEST top-1 remained 45/480, top-5 99→100, and spelling/EOS 2.924645→2.635121. Sampled books previously explored in development; not pristine certification.
- High-strength .50 is a research ablation only: DEV top-1 45→40 (regression). Avoid promoting it as the default despite improved symbol loss and TEST top-5.
- Unchanged original optimizers/checkpoints, exact opt-out v164 rollback, domain isolation (4,545 historical vectors), all historical lineages retained. Full 256-byte BPB, +2/+3, independent pristine TEST, OOV generation beyond 12k fixed words, Android/Termux remain open.

## Historical baseline record (superseded as active)

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


## v163 / v164 trained distributional native continuation — 2026-11-21 (US Pacific)

- **v158 remains active development baseline; v148 is protected fallback. v164 is an opt-in development successor, not a production-certified promotion.**
- v163 cloud-logistic confidence gate was cross-fitted: 400 TRAIN memory books / 100 separate TRAIN calibration books, with 51 book DEV / 64 book TEST held out. Native signed 32D cloud extraction and optimizer compatibility preserved. Cloud-gated top-1 +1/480 DEV but top-5 -1/480; not selected.
- v164 500-TRAIN-book distribution trainer: `experiments/nancm/v164_conditional_train.py`, successful workflow [#44983231506](https://github.com/shemshallah/nanlm_experimental/actions/runs/44983231506); 199,956 context keys and 733,069 candidate entries. Raw index digest `29258157c99d5bdfd5a4fd2622101d2e143f4d893053606612346b6cc8fccd79`.
- v164 C++ native implementation and tests are now in `experiments/nancm/v164/`, with exact blob-SHA parity to the full cumulative source archive, and a normalized nested whole-word mixture (v162 single candidate + v164 up to 16 distributional word candidates), native byte-prefix/EOS conditional API, and fingerprint-locked index.
- Frozen DEV-selected settings: inner v162 memory 0.4, outer v164 distribution 0.6, domain threshold 1.5. 96-beam strict word benchmark: 480 sampled DEV words from 24 books and 480 sampled TEST words from 24 distinct books. DEV v158→v164: top-1 6.25→9.375%, top-5 13.125→17.5%, top-32 23.125→30.4167%. TEST v158→v164: top-1 5.833→9.375%, top-5 15.625→20.625%, top-32 24.375→32.2917%. TEST OOV top32 1→11. Spelling/EOS bits/symbol TEST 3.112825→2.924645; **not full 256-byte BPB**.
- Native v164: 360 exact strength-zero rollback comparisons; all 4,545 historic documentation prefix distributions byte-for-byte identical to v158 due to causal domain gate. Six inherited checkpoint and optimizer artifacts retain exact SHA256. 275 archived files SHA256-checked; fresh clean extraction and C++ recompile, `make test-v164` passed.
- Archive: `NaNCM_v164_DISTRIBUTIONAL_SIGNED_CLOUD_20261121.zip`, 71,159,097 bytes, SHA256 `8094027f605907da7cd439c2ff9b44f50875a5105868a6bbca646045ed7bfb3a`, saved in ChatGPT Library. Only source/test/documentation committed to GitHub; trained checkpoint is in the archive, not in Git.
- Remaining blockers: whole model full-byte BPB and +2/+3 gates; independent fully untouched book test at scale; near-duplicate and inherited pretraining-overlap audit; actual retraining of signed cloud parameters on larger corpus; Android/Termux native tests; strict word top-1 80% target is **not** achieved.
