# NaNCM v174 — Dual-Polarity Causal NOW State

**Research author:** Justin Howard-Stanley · Independent Researcher · Santa Cruz, California, USA.  
**Date:** 2026-11-22.  
**Status:** working native opt-in experiment, **not promoted as active full-decoder**.  
**Single recovery lineage:** inherited v173/v172/v171/v170 checkpoints, optimizer histories and diagnostics retained in a single cumulative recovery ZIP. GitHub remains source-only, `master` branch only.

## Architecture: 128D temporal state coupled to the 128+128 signed cloud

This experiment genuinely trains temporal state, not a continuation-count reranker. The inherited 12,000-word **32D lexical embeddings** and S+/S− **128+128 point geometry** are loaded from their trained checkpoints and frozen. The v174 trainable state has two independently recurrent 64D paths:

- S+ rail consumes 32D causal word embeddings, eight S+ geometric features, eight S− geometric features.
- S− rail consumes the same 32D embeddings, with the eight S− and eight S+ channels interchanged.
- The last causal states `h_plus` and `h_minus` are coupled by `concat(h_plus, h_minus, h_plus * h_minus)` (192D); a learned signed normalization and 192→96→32 projection produces a temporal lexical vector.
- This vector participates in the **normalized 12,000-way exact word probability**: frozen inherited teacher logit + learned temporal energy, scaled by a trainable positive trust parameter. No future words are accessed. A full 128D two-rail temporal NOW state is newly learned.

The inherited v173 6,880-step exact-CE word checkpoint and original optimizer moments are retained; new v174 checkpoint includes **its own three-group AdamW optimizer (15 state entries) and an embedded exact copy of its v173 optimizer state**, and the binary sidecar loads in standalone native C++. No original checkpoints were overwritten.

## Causal data and evaluation scope

Training data were reconstructed from *archived TRAIN books only*: overlapping short passages and correct next-word targets. Deduplicate by `(book, 12-word causal context, target)`. 78 TRAIN books supplied **11,057** unique word-context records; another 20 excluded TRAIN books supplied **2,887** word-context validation records. These 20 books were *not used to train the new temporal parameters*, but may occur within the inherited v173 teacher's earlier aggregated pretraining. This is a paired book-group validation of newly trained parameters **not independent whole-system book-disjoint certification**.

Trained 300 updates, then resumed that checkpoint **including its optimizer state** for 1,400 further updates. Selected **step 800** using the heldout TRAIN-book validation *cross entropy* (not DEV). Subsequent overtraining worsened the validation loss; the selected checkpoint and its exact Adam moments were preserved.

| Result | Frozen v173 word expert | v174 signed temporal expert |
|---|---:|---:|
| TRAIN-book-heldout next-word NLL (nats) | 6.091558 | **5.997822** |
| TRAIN-book-heldout top-1 / 2,887 | 305 | **333** |
| TRAIN-book-heldout top-5 / 2,887 | 675 | **729** |
| Exposed DEV in-vocab NLL, n=227 | 6.525258 | **6.492202** |
| Exposed DEV standalone top-1 / 227 | 14 | **18** |
| Exposed DEV standalone top-5 / 227 | 42 | **44** |

Book-block bootstrap on the 20 TRAIN-book holdout groups, resampling 5,000 times: 95% interval for NLL improvement **[.07295, .11566]**; top-1 *percentage-point* improvement **[0.173, 1.954]**. These intervals measure this archived-book sample only.

An entirely causal whole-word expert fusion was calibrated **only on 196 TRAIN-book contexts**. It improved full-decoder TRAIN-book calibration **27→30** top-1, but **regressed the historically exposed DEV whole-word top-1 from 27/244 to 26/244**, and top-5 from 57 to 56. **Fusion rejected; no active word promotion**. The unchanged v167 active word baseline is protected. Open-vocabulary success and 80% next-word accuracy are not claimed.

## Native tests and preservation

- Pure native 12,000-way C ABI (including native 12-token causal tokenization and dual-GRU inference), trained sidecar with three inherited cloud fingerprints, tensor-shape and finite-value validation, fail-closed malformed inputs.
- Native/PyTorch **74 full distribution** parity checks: 12,000 probabilities normalized; maximum elementwise difference 2.086e-7; exact top-1/top-5 agreement.
- Inherited native word rollback **240 exact 27-way prefix+EOS vectors**, 48 normalized byte-to-word root agreement calculations and **144 exact later-prefix** vectors.
- Inherited v170 byte streaming replay **8,015 DEV + 8,003 reused TEST** symbols: 292 exact 256-way v169 rollback vectors, unchanged 2.5638488935 DEV BPB and 2.4727568337 reused-TEST BPB.
- Standalone source-only C++ self-test synthesizes a fresh valid 12,000-word sidecar and asserts normalized prediction, invalid/short argument rejection, malformed magic fail-close. This test requires no trained binaries and is runnable on GitHub Actions.

The v173 source and native ABI are inherited unmodified by `experiments/nancm/v174/wordfield174_api.cpp`, which includes v173 source before the new temporal ABI. The opt-in v174 calls are `wf174_now_open`, `wf174_now_predict`, `wf174_now_word_name`, `wf174_now_word_count`, `wf174_now_close`.

## Reproduction (from complete recovery ZIP)

```sh
make -B v174
OPENBLAS_NUM_THREADS=1 python experiments/nancm/v174/test_native_temporal174.py --full
OPENBLAS_NUM_THREADS=1 python experiments/nancm/v174/test_rollback_word174.py
OPENBLAS_NUM_THREADS=1 python experiments/nancm/v174/test_byte_rollback174.py
OPENBLAS_NUM_THREADS=1 python experiments/nancm/v174/eval_book_holdout174.py
OPENBLAS_NUM_THREADS=1 python experiments/nancm/v174/eval_temporal_fusion174.py
```

To continue training **without resetting the newly learned optimizer**, run `python experiments/nancm/v174/train_temporal_signed174.py --resume --steps 1000`. Selection still occurs on TRAIN-book validation loss. The sidecar must be repacked with `pack_temporal_signed174.py` if a new checkpoint is explicitly selected; no direct overwrite of inherited parent model is performed.

## Limitations / next step

The corpus snippets cover at most a few hundred history bytes per archived TRAIN example, not entire books. This is a larger, recurrently trainable temporal model, but **not the full-book, truly unseen-corpus leap** needed for high word accuracy. True generalization requires acquiring the NaNROUTER Releases book corpus, preserving book-disjoint train/validation/test hashes, and training on full causal passages. No fresh release dataset was downloaded in this round; all historic DEV and TEST windows are already exposed.