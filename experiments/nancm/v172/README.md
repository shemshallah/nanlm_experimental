# NaNCM v172 — signed spatiotemporal rails + trainable native whole-word teacher

**Author:** Justin Howard-Stanley · Independent Researcher · Santa Cruz, California, USA  
**Date:** 2026-11-22  
**Status:** OPT-IN RESEARCH ONLY. Active whole-word development baseline **v167**; strongest previously measured active byte experiment **v170**. No trained parent checkpoint or optimizer history was overwritten. `master` only; source folder `experiments/nancm/v172`.

## Architecture (one inherited system, two experimental integrations)

1. **Causal spatiotemporal signed-cloud retrieval:** inherit the 128 lower + 128 upper S+/S− cloud, v166 lexical embeddings, v168 199,956-address ANN, and v171 32–512 neighbor / 64–1,024 candidate bounds. `V172_ANN_RAILS` enables 1–12 deterministic retrieval rails. Rails 1–6 vary the signed/lexical projection ratio while using the same three-word causal context. Rails 7–9 use the last completed word; rails 10–12 use the last two completed words. Every rail contributes a normalized count distribution. **Default is 1 rail**, exactly the v171 path; no target/future bytes enter a query. At 512 neighbors and 1,024 candidates per ranker, ANN top-1 and recall on the 244 exposed gate-active DEV contexts are: 1 rail **20 / 165**, 6 rails **20 / 179**, 12 rails **19 / 178**. Combined native/ANN candidate recall at 6 rails is 179/244. This is *candidate coverage*, not a first-choice decoder result. Full 96-beam word decoding remains 27/244 top-1 with 6 rails and 12 rails. 

2. **Trainable conditional signed-cloud word expert:** inherited 32D lexical tensors, 128×128 S+/S− signed cloud and 128+128 geometric points are used to initialize a learned 8D→32D geometric linkage. Trained on **692,374 available TRAIN-derived n-gram conditional entries**, filtered to the inherited 12,000-word vocabulary, with sampled noise-contrastive loss and preserved AdamW moments. Step-480 checkpoint was continued, not reset; **step 6,000** selected by predesignated TRAIN context-key heldout loss. PyTorch checkpoint `checkpoints/v172_conditional_signed_cloud.pt` includes all optimizer param groups and moments. Native sidecar `checkpoints/v172_native_signed_teacher.bin` (`N172TCH1`) is fingerprint-locked to inherited model, geometry, and cloud; malformed/missing sidecars fail closed. The 12,000-way native probability vector matches PyTorch within ~1.04×10⁻⁷ on 21 exposed DEV contexts.

The new whole-word teacher participates in the **exact normalized distribution**:

`P172(w | history) = (1 − gamma) * P171(w | history) + gamma * Q172(w | history)`

via a 27-way prefix/EOS native interface and inherited open-vocabulary residual, not a heuristic candidate-score reordering. Tested on 196 complete word probability identities (max |difference| < 2.3×10⁻⁹). `V172_TEACHER_SHARE` defaults to 0, preserving the older word/byte API exactly; set >0 with `V172_TEACHER_MODEL=<fingerprinted sidecar>` for experimental inference.

## Matched results and rejection decision

| Experiment | Cohort | Existing parent | v172 | Decision |
|---|---|---:|---:|---|
| Signed ANN candidate recall, 512 neighbors | DEV gate-active 244 | 165/244 single rail | 179/244, six rails | Keep as opt-in retrieval |
| Full 96-beam next-word top-1 | same DEV 244 | 27/244 | 27/244 six rails | No word promotion |
| Word NLL, pretrained vs trained teacher | reused DEV 227 in-vocab | 11.7240 nats | 6.5282 nats | Real expert learning, not full decoder improvement |
| Standalone teacher top-1 | same DEV 227 | 11/227 pretrained | 16/227 trained | Below protected decoder |
| Trained teacher native mixture gamma=0.40, **TRAIN calibration** | 180 TRAIN-book contexts | 30/180 | 32/180 | Select gamma=.40 inside TRAIN only |
| Frozen gamma=.40 native mixture, **exposed DEV** | 244 contexts | 27/244 | **21/244** | Reject opt-in teacher as word default |
| Native next256 default rollback | DEV 8,015 bytes | 2.563848894 BPB | 2.563848894 BPB | Exact preservation |
| Native next256 default rollback | reused TEST 8,003 bytes | 2.472756834 BPB | 2.472756834 BPB | Exact preservation |

`TRAIN` word calibration: 71 fit books / 20 holdout books, 670 / 180 contexts. The direct context-aware portfolio reranker reached 34/180 TRAIN validation top-1 vs 30 parent, but **27/244 vs 27** on DEV, not promoted. An earlier residual linear scorer fell to 21/244 DEV; rejected. A train-calibrated heuristic on the 6k-step teacher fell to 25/244 DEV; rejected.

**Critical scientific caveats:** the n-gram table was preaggregated over 500 TRAIN books and has lost book identifiers. The teacher's 5% deterministic context-key holdout is **not book-disjoint** and can contain information from the same books as training. The DEV and TEST windows have previously been accessed, and the retrieved ANN evaluation includes exposed gate-active positions only. Neither 80% strict next-word accuracy nor untouched book generalization is established. Android parity and +2/+3 true lookahead remain open.

## Native reproduction from the cumulative recovery ZIP

```sh
OPENBLAS_NUM_THREADS=1 make -B v172
OPENBLAS_NUM_THREADS=1 python experiments/nancm/v172/test_native_teacher172.py
OPENBLAS_NUM_THREADS=1 python experiments/nancm/v172/test_teacher_wholeword_mixture172.py
OPENBLAS_NUM_THREADS=1 python experiments/nancm/v172/test_byte_parity172.py
OPENBLAS_NUM_THREADS=1 python experiments/nancm/v172/bench_teacher_word172.py --split dev --share .4
```

For exact v171 word/byte rollback set `V172_ANN_RAILS=1 V172_TEACHER_SHARE=0`. For widened retrieval set `V172_ANN_RAILS=6 V171_ANN_NEIGHBORS=512 V171_ANN_CANDIDATES=1024`, leaving the teacher strength at 0. For native teacher only, set `V172_TEACHER_SHARE=.4 V172_TEACHER_MODEL=checkpoints/v172_native_signed_teacher.bin`; this research profile is **rejected as active default** by the DEV result. Both new pathways are opt-in and retain optimizer lineage.

GitHub contains source and smoke tests only; the full restore archive includes original trained models and optimizer checkpoints. CI does not claim model training parity without the archive.