# NaNCM v173 — Exact-CE Signed Cloud + Causal Byte/Word Agreement

**Author:** Justin Howard-Stanley — Independent Researcher, Santa Cruz, California, USA.  
**Date:** 2026-11-22.  
**Promotion:** NO. Parent v167 whole-word and v170 byte-lineage are preserved.

## Architectural changes (symbiotic, native, and strictly causal)

- Continued the **one inherited trained 12,000-word 32D lexical head** on the 128+128 S+/S− signed geometry. Switched objective from noise-contrastive estimation (NCE) to **exact normalized whole-vocabulary 12,000-way softmax cross-entropy**. AdamW moments resumed from v172 step 6,000; selected experimental step 6,880 on SHA context-key TRAIN-derived validation only. The optimizer still has 3 parameter groups, 5 nonempty parameter-state slots. The old 6,000-step checkpoint and all previous optimizers remain intact. Native sidecar uses fingerprint-checked N172TCH1 codec, accepted without changes to v172 native teacher.
- Introduced `wf173_joint_open`, `wf173_joint_begin`, `wf173_joint_next`, `wf173_joint_close`. These combine the **causal native 256-way byte predictor** and 27-way whole-word model by a **geometrically normalized first-letter product** in the 26-simplex, preserving the inherited EOS probability and the remaining conditional word suffix model. Gamma=0 gives **bitwise exact v172 rollback** and never requires loading a byte pack. Native code is compiled as `bin/libwordfield173.so` from `experiments/nancm/v173/wordfield173_api.cpp`.
- First-letter joint gamma=.10 chosen on **TRAIN-book calibration**, **not DEV**; exposed DEV used only for the final reported comparison. The native byte model's prequential adaptation is disabled for these independent random-access word contexts; it is still active and regression-tested in the original streaming byte API.

## Matched measured results

| Metric | Inherited parent | Experimental v173 | Interpretation |
|---|---:|---:|---|
| Full 96-beam word top-1, 197 TRAIN-book calibration | 23 | 24 | Modest TRAIN-only gain |
| Full 96-beam word top-1, 244 historically exposed DEV contexts | **27** | **27** | **No strict word gain** |
| Word top-5, same DEV | 57 | 57 | No gain |
| Word top-32, same DEV | 98 | 98 | No gain |
| First-letter conditional NLL, 244 exposed DEV word starts | 2.929150 | **2.895578** | Better first-letter calibration only |
| First-letter top-1, same DEV | 43/244 | 43/244 | No accuracy gain |
| Full v170 next-byte DEV BPB, 8,015 bytes | 2.563848894 | **2.563848894** | Exact retained |
| Full v170 reused TEST BPB, 8,003 bytes | 2.472756834 | **2.472756834** | Exact retained |

The v173 12,000-word teacher exact-CE model reduced the *reused DEV* in-vocabulary NLL from **6.52818** to **6.52526 nats** (227 words), but its standalone top-1 **fell from 16 to 14/227** and top-5 from 43 to 42/227. On the TRAIN-derived non-book-disjoint 1,536-context validation sample, NLL fell from **6.263737** at step 6,000 to **6.260751** at selected step 6,880. It remains **experimental**, never replacing v172's stronger standalone top-1.

## Negative controls and independent subtasks

- Native default-disabled v173 reproduced v172 **240 exact 27-way prefix/EOS vectors**. Enabled root calculations passed 48 independent normalized mixture identities with largest |difference| **1.42e-08**. Another 144 later-prefix vectors matched v172 exactly. Missing sidecar and invalid gamma fail closed.
- Native exact-CE teacher was matched to PyTorch in 21 contexts, max probability difference **5.96e-08**, all 12,000 probabilities normalized.
- Entire native v170 256-byte streaming replay passed on 8,015 DEV + 8,003 reused TEST bytes; 292 entire 256-way v169 rollback vectors had max absolute difference **0**. No trained parent/optimizer changes.
- Rejected candidate-list debiaser: trained on 640 positions and selected on 210 TRAIN-book positions, selected method yielded **24 / 244** DEV top-1 vs 27 parent.
- Rejected causal 1–3-gram mixture: TRAIN-book validation 32→36/197 but DEV **18/244** vs 27 parent.
- Native byte-first-letter top-1 was **45/244** across word starts; TRAIN-selected fusion on archived candidate lists yielded **27/244**, tied with 27 baseline. Parent root first-letter top-1 43/244, joint 43/244, though root NLL improved.

## Limitations and promotion gate

Word starts here are drawn from **previously explored DEV books**; historical TEST has also been used. The 692,374 TRAIN-derived n-gram association rows lost per-book attribution in aggregation. The exact-softmax checkpoint was therefore selected on an SHA context-key holdout, **not** a book-disjoint validation. Train-book selection of the root gate is disjoint from its own TRAIN examples, but those books still may overlap preaggregated cloud training. No untouched book-disjoint certification or 80% strict word top-1 result is claimed.

**Do not promote.** Both improvements were confined to calibration/NLL. Achieving a strict whole-word breakthrough requires a causal sequence model trained directly on full-book passages and evaluated on genuinely unseen books, likely with a hierarchical byte/BPE/word/sentence stack rather than a shallow mixture of existing experts.

## Reproduction from complete v173 recovery ZIP

```sh
make -B v173
OPENBLAS_NUM_THREADS=1 python experiments/nancm/v173/test_native173.py
OPENBLAS_NUM_THREADS=1 python experiments/nancm/v173/test_native_teacher173.py
OPENBLAS_NUM_THREADS=1 python experiments/nancm/v173/audit_root173.py
OPENBLAS_NUM_THREADS=1 python experiments/nancm/v173/bench_joint173.py --cohort dev --gamma .1
OPENBLAS_NUM_THREADS=1 python experiments/nancm/v173/test_byte_parity173.py
```

Full training, optimizer history, signed ANN, C++ shared library and byte sidecars are included only in the cumulative recovery ZIP. GitHub `master` is **source-only**, with Actions smoke tests; numerical replay is performed against the complete archive.