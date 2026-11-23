# NaNCM — Active development baseline v167

**2026-11-22: v167 Agreement-Gated Signed Lexical Birth is the active DEVELOPMENT baseline.** This supersedes the user-promoted v166 .05 conservative profile, not its protected neural checkpoint. v167 still loads the exact same 12,000-word v166 checkpoint and full AdamW history, inherited from v165 128+128 signed cloud. It changes only causal admission and consensus-controlled lexical mixing; no original model weights are overwritten.

Frozen profile: `V166_LEXICAL_BIRTH=.05 V166_BIRTH_MODEL=checkpoints/v166_lexical_birth.bin V167_GATE_MODE=1 V167_HIGH_ALPHA=.5 V167_TRUST_THRESHOLD=.5 V167_DOMAIN_MIN=1.4`. Native `bin/libwordfield167.so` and unchanged `wf148_*` ABI. Source/tests in `experiments/nancm/v167/`; cumulative recovery ZIP includes native trained sidecars and original optimizer checkpoints.

DEV 480 independent-book sampled words: v166→v167 strict top-1 **45→46**, top-5 **86→94**, top-32 **146→150**, spelling/EOS bits/symbol **2.624520→2.558560**. Reused TEST sample: top-1 **45→46**, top-5 **100→104**, top-32 **159→162**, bits/symbol **2.635121→2.512776**. v167 enables generator at 244 DEV positions vs 219 for v166; 951 exact v166 rollback vectors and 4,545 docs-domain vectors pass. Top-1 lift is only 1/480 and confidence intervals are wide; **not production-certified**. Both book samples reused, not pristine. Full 256-byte BPB, +2/+3, Android, overlap audit, 80% word top-1 remain OPEN.

Fallback lineage: v166 conservative .05 → v164 → v158 preserved checkpoint → v148 legacy fallback. The additional v167 margin-trained checkpoint is archived as rejected because book DEV top-5 regressed. No new branches; master only.

## Historical promotion records

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


## v169 Native Symbiotic 256-Byte experiment — 2026-11-22

**Active development word baseline remains v167.** v169 is research-only on master, not an active checkpoint promotion. It composes inherited 128+128 signed S+/S− cloud, v166 trained lexical head and v168 signed ANN with a new native 256-way byte 0–3gram and word bridge. Its frozen DEV-selected mixture records full next-byte BPB **2.587376** on 8,015 DEV bytes and **2.505834** on 8,003 previously reused TEST bytes; 24/24 book windows improve versus a same-TRAIN fitted byte ngram. These figures belong to the NEW hybrid byte predictor, not historical 27-way spelling/EOS BPB. Strict word DEV top-1 still **46/480** and **141/1,394**, without advancement. No claim of untouched test, Android readiness, or 80% word top-1. Source and regression under `experiments/nancm/v169/`; trained sidecars and entire optimizer/checkpoint lineage only in complete recovery ZIP.

## v171 signed ANN neighbor-width experiment — 2026-11-22

Opt-in native v171 source under `experiments/nancm/v171/`, **not promoted**. Matched exposed DEV 244 ANN-active contexts: 32/64 ANN recall 93, widened 512/1024 recall 160; native + lexical + widened ANN union ceiling 167. Full 96-beam strict word top-1 stays 27/244 across width arms. v170 16,018-byte 256-way replay unchanged; v167 active word model protected. Latest complete cumulative checkpoint held in v171 ChatGPT Library ZIP, not source-only Git. No new optimizer lineage or branch.

## v172 signed multi-rail + trained 32D word teacher — 2026-11-22

Native source, train, optimizer resume, exact 12k-word mixture and 256-byte rollback scripts are in `experiments/nancm/v172/`. New opt-in retrieval reaches ANN candidate recall 179/244 at 6 rails × 512 neighbors on exposed DEV, but strict 96-beam word top-1 remains 27/244. The signed geometric conditional word expert was trained to step 6000 on 692,374 TRAIN-derived conditional entries with AdamW history preserved. Frozen .40 native teacher mixture selected on TRAIN improved calibration 30→32/180 yet regressed exposed DEV 27→21/244. **Not promoted.** Active word v167 and experimental byte v170 remain unchanged. Cumulative trained checkpoints/optimizers only in latest Library recovery ZIP; no fresh-book certification.

## v173 exact-CE signed cloud + normalized byte/word first-letter gate (2026-11-22)

Cumulative trained recovery checkpoint available in current NaNCM Library ZIP only. This v173 source folder adds a **new 12k exact-softmax CE continuation at step 6,880**, retaining the v172 step 6,000 optimizer lineage and inherited 128+128 signed geometry. Native opt-in `wf173_joint_*` fuses full 256-byte first-letter support with parent 27-way word prefix probabilities; gamma=0 has exact v172 rollback. Full native byte 16,018-byte replay unchanged. 96-beam strict word DEV top1 **27/244 vs 27/244 parent**, no default promotion; first-letter NLL 2.92915→2.89558 on exposed DEV. Protected active word baseline v167, native byte v170. Training holdout context-key not book disjoint; DEV and TEST already exposed. This repository is source-only; trained weights and 3-group optimizer checkpoint remain in cumulative recovery ZIP. No other branch created.

## v174: frozen-cloud dual signed temporal NOW state (2026-11-22)

Source-only v174 in `experiments/nancm/v174/`. Existing v173 12k-word 32D exact-CE teacher is preserved, plus inherited 128+128 S+/S− geometry and full optimizer. New 64+64 causal GRU NOW-state and 12k normalized temporal residual, selected step800 of 1700 attempted updates, preserve new 3-group optimizer history and inherited teacher optimizer in latest cumulative Library checkpoint. Book-separated 20 TRAIN-books (not pristine end-to-end) n=2887, NLL 6.09156→5.99782, top1 305→333, top5 675→729. Exposed DEV expert top1 14→18/227, but full decoder TRAIN-tuned fusion regressed DEV 27→26/244, so **no active word promotion**. Native C++ verified 74 whole 12k distributions vs PyTorch, 240 exact native root rollback, 292 256-way byte rollback, 16018 byte replay unchanged. Retain v167 active word, v170 byte baseline. Full recoverability/weights/optimizer only in single v174 Library recovery ZIP; GitHub workflow source-only.

## v175 causal 48-word signed temporal copy expert — source and fresh cohort

New source under `experiments/nancm/v175/` continues the **v174 64+64 S+/S− recurrent NOW state** with its exact three AdamW optimizer groups and adds a fourth trainable causal repeated-prefix pointer. The pointer predicts only *earlier observed continuations* after an occurrence of the current word/phrase. Native v175 loads a 48-history signed GRU and 12,000-way exact-softmax sidecar. Inherited public v174/v170/v167 ABIs remain compiled and default-disabled. Source-only GitHub Actions prepares NaNROUTER Release `corpora-short-stories-0003` and has produced 500 book-disjoint selected books: 396 TRAIN (95,040 examples), 55 validation (13,200), 49 TEST (11,760). The 615 known inherited cohort book IDs are excluded. Source train workflow and optimizer lineage are recoverable only with the latest cumulative Library ZIP, not the source-only GitHub repo. v174 active word/byte promotion rules remain; v175 full-decoder mixture not promoted. The 49-book TEST was first inspected after the 280-step pilot and must be labeled reused on later rounds; all later selection uses TRAIN-book validation only. Complete numerical benchmark and claim ledger in the release archive.

## v175 final selected 2026-11-22 — strongest standalone 12k signed temporal specialist

Best checkpoint selected using 55-book validation NLL after 5,080 attempted v175 updates: **v175 step4,955** (total step5,755 with inherited v174). The four-group optimizer (19 state entries) and parent v174 and v173 exact optimizer histories are preserved in the sole Library recovery ZIP. New NaNROUTER Release0003 396/55/49 book-disjoint split, known old 615 book IDs excluded. On 49-book now-reused TEST with in-vocab targets n=11,760, frozen v174 word NLL **6.13305** vs v175 **6.00422**, top1 **1,242→1,364**, top5 **2,918→3,123**. Repeat-copy ablation top1 1,325→1,364. 3,000 book-bootstrap 95% NLL gain **[0.1113,0.1463]**. Provenance caveat: TEST inspected after earlier v175 pilots; inherited teacher pretraining not fully excluded; standalone 12k conditioned accuracy only. 80% accuracy not attained. **Promote only as standalone signed temporal research specialist; default v167 open-vocab word path and v170 byte path stay protected**, no claim of full-decoder closure. Native 80 full 12k float-parity max abs 5.07e-7, 25 exact old native word ABI, 292 exact byte rollbacks and 16,018-byte replay unchanged. Next goal counterfactual trust gate to reject wrong repeated-prefix copies. Source-only CI passed.
