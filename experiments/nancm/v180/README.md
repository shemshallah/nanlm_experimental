# NaNCM v180 — Book-disjoint causal word memory, normalized signed-cloud fusion

**Author:** Justin Howard-Stanley · Independent Researcher · Santa Cruz, California, USA. 2026-11-23.

**Goal:** advance strict next-whole-word top-1 toward 50% on independent books. **Result:** measurable progress, far below the target; experimental only. Previous v179 clone-free COW multi-context and its exact byte law are inherited without alteration. `master` only.

## What changed

- The v175 Release0003 TRAIN split had 95,040 labeled 48-word windows over 396 books. Using each row's **word position** and already-observed context, `prepare_stream180.py` reconstructs 3,115,680 observed TRAIN-word positions (including 223,910 OOV markers), with **1,541,280 overlapping positions agreeing exactly and zero conflicts**. Gaps between samples remain `-2`, OOV remains `-1`; neither can bridge an n-gram. 2,650,489 valid one-word-prefix transitions result. **No validation or test book enters the word-memory index**.
- Pure C++ exact-key 1–4 word suffix tables are fit to this recovered TRAIN stream: 10,620 / 525,966 / 1,453,295 / 1,870,824 distinct context keys, respectively. 14-bit symbols pack exactly into 64-bit context keys for n≤4. Tables retain exact top-two continuations and counts; they do **not** invent a complete n-gram probability distribution for unobserved words.
- The **same inherited** 128+128 S+/S− signed-cloud v176 whole-word probability vector is computed first. Each suffix expert contributes a **mathematically normalized sparse count residual** with globally TRAIN-selected weights `gamma=[0.1,0.1,0.1,0.1]`:

  `P180(w|h) = [1 - Σ_n gamma_n M_n(h)] P176(w|h) + Σ_n gamma_n c_n(w|h)/C_n(h)`

  where `M_n=(top1count+top2count)/C_n`, `c_n` is zero except for the exact top two, and an absent key contributes zero. Thus every distribution remains nonnegative and sums to one. This is an opt-in word probability model; it does not modify the inherited WF170 byte law.
- Numba/LGBM 38-feature candidate ranking was also trained as a parallel track and **rejected** because it generalized worse. A top-1-optimized TRAIN calibration was also rejected versus the NLL-selected weights.
- `online_memory180.py` evaluates an additional within-book prequential cache that only reads observed history and previous revealed labels. It was not fused into the accepted predictor because its effect as a full model has not been established.

## Exact strict in-vocabulary word evaluation (55 reused but book-disjoint VAL books)

| Metric | Inherited v176 | v180 normalized mixture | Gain |
|---|---:|---:|---:|
| Top-1 (n=13,200) | 1,687 / 12.78% | **1,893 / 14.34%** | +206; +1.56 pp |
| Top-5 (n=13,200) | 3,826 / 28.98% | **4,161 / 31.52%** | +335; +2.54 pp |
| Word NLL (nats/word) | 5.81715995 | **5.71369394** | −0.10346601 |
| Native 12k-vector parity (all 13,200 VAL contexts) | parent preserved | PASS | max error 9.54×10⁻⁸ |

The paired **whole-book** bootstrap (3,000 replicates, 55 books) estimates top-1 gain 95% interval **[+1.21, +1.90] percentage points**, and NLL reduction **[0.0900, 0.1168] nats/word**. These uncertainty intervals do not correct for repeated use of this VAL set.

**Ceiling diagnosis:** at most 8,370/13,200 (63.41%) correct targets appear in inherited neural top-64 unioned with top-two suffix continuations, before prequential local history. Adding the strictly causal within-book local cache increases this to 8,541/13,200 (64.70%) but was not integrated. Reaching 50% strict top-1 requires a much stronger generalizing ranker/base model, not simply speculative decoding.

## Critical limitations

1. All targets are within the fixed 12,000-word vocabulary. **Out-of-vocabulary next words are excluded.** These scores do not certify free-form open-vocabulary generation.
2. Validation books are distinct from the new 396 TRAIN books, but this 55-book cohort has been inspected repeatedly since v175, and the inherited teacher's older pretraining corpus may overlap. This is a **diagnostic, not a pristine external validation**. No 49-book TEST labels were consulted in v180.
3. Word NLL is genuine normalized 12k word NLL; byte BPB, +2/+3 predictions, held-out natural-text decoding accuracy, and end-to-end sampling speed have **not** been improved or remeasured.
4. v180 table construction currently costs ~2–3 CPU seconds and holds several million exact keys in RAM. Android device-specific memory/throughput is untested. C++ API is opt-in with borrowed cross176 parent handle; close memory before closing parent.
5. The LGBM ranker and TRAIN-top1 calibration are retained in the experiment records as **rejected**, not deployed.

## Reproduce

```sh
python experiments/nancm/v180/prepare_stream180.py --split train
python experiments/nancm/v180/prepare_rows180.py
make -B v180
make v180-source-smoke
./bin/nancm_v180_memory results/v180/train_observed_stream.bin results/v180/train_rows.bin results/v180/val_rows.bin
OPENBLAS_NUM_THREADS=1 python experiments/nancm/v180/collect_neural180.py --split train
OPENBLAS_NUM_THREADS=1 python experiments/nancm/v180/collect_neural180.py --split val
OPENBLAS_NUM_THREADS=1 python experiments/nancm/v180/normalized_mix180.py
OPENBLAS_NUM_THREADS=1 python experiments/nancm/v180/frozen_book_eval180.py
OPENBLAS_NUM_THREADS=1 python experiments/nancm/v180/test_native180.py --full
```

Reconstruction only permits `--split train`. It fails on word overlap conflicts. `native_memory180.cpp` is a standalone diagnostic; `libwordfield180.so` is the integrated exact word-law API. Inherited checkpoints, AdamW moments, v179 fork/COW and v170 256-byte tables remain untouched. Native API: `wf175_now_open` → `wf176_cross_open` + `wf176_cross_add_bpe` → `wf180_memory_open(cross,train_stream)` → `wf180_memory_predict` → `wf180_memory_close` → parent close.

## Promotion

v180 is the **best measured opt-in 12k conditional whole-word specialist** on reused Release0003 VAL. It is **not** promoted to the active unrestricted model, single-stream v179 verifier, or native byte decoder. The 50% strict top-1 objective remains open and requires more powerful training on larger book-disjoint corpora, improved candidate recall, and a sealed unseen test cohort.
