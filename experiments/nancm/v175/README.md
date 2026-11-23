# NaNCM v175 — Full-book causal signed temporal + episodic word pointer

**Author:** Justin Howard-Stanley — Independent Researcher, Santa Cruz, California, USA  
**Date:** November 22, 2026  
**Research status:** new best **experimental standalone 12,000-word temporal specialist**, **not** promoted to the default unified open-vocabulary word decoder. `master` only; original checkpoints and optimizer states preserved.

## Architecture (symbiotic, not an independent reranker)

The trained 128+128 S+/S− concept cloud and v173 32D lexical base are inherited. The previously trained v174 dual 64D GRUs now ingest **48 whole-word causal positions** (instead of 12). We continue **all original three AdamW groups**, including their actual moments; a **fourth optimizer group** learns a repeated-prefix causal pointer with four trainable parameters.

For each prior time index `i < t`, when the earlier word `w_i` equals the most recent word `w_t`, the *already observed* following word `w_(i+1)` receives an energy boost. Match agreement over the previous two or three words amplifies the boost. The learned strength and decay are normalized by the amount of historical evidence. These energies are added to the inherited signed-GRU logits and normalized over all **12,000 complete words**, so neither lookahead nor unnormalized reranking is involved. There is **no copy from the target or from future words**. The native `N175NOW1` binary encodes the 24 inherited tensors plus four new trainable parameters, tied to three inherited cloud fingerprints.

## Real NaNROUTER Release data, new training split

Data from [NaNROUTER short-story batch 0003](https://github.com/shemshallah/NaNROUTER/releases/tag/corpora-short-stories-0003) with verified archive SHA256 `15e7b5937a8ae8f5dc41f9f4b650202f9a88fb2a299daee38653c6a3d4ba8a20`. GitHub Actions downloaded the original **91,625,172-byte** archive, verified the Release's checksum, streamed entire books, and saved **48-word causal windows across their full lengths**. The exact 12k vocabulary order, 615 old book-ID exclusion list, per-book content hashes, all split definitions, and compact fixed numerical examples are included in this release.

| New data partition | Books | Examples |
|---|---:|---:|
| TRAIN | 396 | 95,040 |
| Validation | 55 | 13,200 |
| TEST | 49 | 11,760 |

**Book identities are disjoint within this new split**. Known 615 inherited book IDs were excluded; no selected book shared one of those IDs. We **cannot certify** that the much older inherited neural teacher never encountered these texts through other pretraining. Word accuracy and NLL are **conditional on targets present in the 12k vocabulary**. Some targets are out-of-vocabulary and excluded. The first TEST inspection occurred at v175 step280; subsequent results on this TEST are **reused** and not untouched certification. Checkpoints are selected using TRAIN-derived validation loss, not TEST.

## Selected training checkpoint

Four CPU stages, preserving both trained tensors and AdamW moments each time: 280 + 1,200 + 1,600 + 2,000 = **5,080 attempted new updates**. The selected **v175 update 4,955** achieved lowest validation NLL. Including inherited v174 step800, the stored step counter is **5,755**. All four current optimizer groups (19 state entries) and exact v174/v173 parent optimizer histories are recoverable. Frozen signed geometry and lexical embeddings are byte-exact preserved; temporal GRUs, residual projection and pointer weights changed measurably.

## Matched strict whole-word scores (in-vocabulary only)

| Metric | Frozen v174 (12 words) | v175 (48 words + pointer) |
|---|---:|---:|
| 55-book validation NLL ↓ | 6.070959 | **5.939311** |
| Validation top-1 | 1445/13,200 | **1573/13,200** |
| Validation top-5 | 3307/13,200 | **3596/13,200** |
| Reused 49-book TEST NLL ↓ | 6.133046 | **6.004215** |
| TEST top-1 | 1242/11,760 | **1364/11,760** |
| TEST top-5 | 2918/11,760 | **3123/11,760** |

Gain in TEST NLL = **0.128830 nats/word** (~2.10%). TEST top-1 gains **122 words**. Across 3,000 49-book block-bootstrap resamples, conditional 95% interval for NLL gain: **[0.1113, 0.1463] nats**. This is evidence of a positive diagnostic; the reused test and incomplete inherited pretraining provenance prevent pristine generalization certification.

### Causal pointer mechanism and unrelated-history control

In TEST, a prior occurrence of the latest word existed for **3490/11760** contexts; the target was among the previously observed following words in **399** contexts. Turning OFF the copy term at the selected same weights changes strict TEST top-1 from **1364 to 1325** and NLL from **6.004215 to 6.027954**. In 399 directly copyable contexts, top-1 improved from **106 to 190**, but wrong-copy contexts regressed from **225 to 180**. This exposes a concrete next research target: **train a counterfactual trust gate on repeat reliability**, instead of increasing unconditional copy gain.

A causal negative control replaces older words with history from a different book while preserving the last three words. On 1,500 test positions, correct older history has **0.0410 nats** lower word NLL than unrelated older history. This is a *distribution-shift diagnostic*, not a causal proof.

## Native C++ and inherited regression

- The full native 12,000-way v175 distributions match PyTorch on **80** examples: max abs error **5.07e-07**, exact top-1 and top-5 order.
- **25** full 12,000-way legacy v174 rollback distributions are **bit-exact**, evaluated through the compiled v175 library.
- The fully inherited prequential byte decoder scored **16018** native positions and **292** exact 256-way fallback vectors. Byte BPB unchanged: DEV **2.563849**, historically reused TEST **2.472757**. Earlier DEV/TEST byte cohorts are not fresh.
- Standalone GitHub source-only C++ test constructs and corrupts a synthetic 12k sidecar and checks normalized probabilities, invalid arguments, and causal copy. Source CI cannot load the trained archive because GitHub is intentionally source-only.

## Reproduce from this recovery ZIP

```sh
# Full local archive already contains inherited model, optimizer and dataset.
python -m pip install torch numpy
make -B v175
OPENBLAS_NUM_THREADS=1 python experiments/nancm/v175/test_native175.py --n 80
OPENBLAS_NUM_THREADS=1 python experiments/nancm/v175/test_byte_rollback175.py
OPENBLAS_NUM_THREADS=1 python experiments/nancm/v175/eval_book_disjoint175.py --split val
OPENBLAS_NUM_THREADS=1 python experiments/nancm/v175/eval_book_disjoint175.py --split test
OPENBLAS_NUM_THREADS=1 python experiments/nancm/v175/audit_causal_echo175.py --split test
```

For a further **optimizer-preserving** training continuation, use `python experiments/nancm/v175/train_long175.py --dataset results/v175_release3 --resume --steps 1000 --skip-test`. Do not train against TEST and do not replace the frozen v174 checkpoint. Source-only GitHub `master` commits: [corpus `65d58fe`](https://github.com/shemshallah/nanlm_experimental/commit/65d58fe772320c9d563a6fd8cf1f9fcca6a1dc4f), [native/training `d36feab`](https://github.com/shemshallah/nanlm_experimental/commit/d36feab0073faba308a680bf69cee382ecf94871); [dataset Actions #45094346173](https://github.com/shemshallah/nanlm_experimental/actions/runs/45094346173); [source smoke Actions #45095193869](https://github.com/shemshallah/nanlm_experimental/actions/runs/45095193869).

## Promotion and limitations

**Yes:** select v175 as the best standalone experimental signed-temporal 12k-word specialist on the new validation metric. **No:** do not promote to the active unified v167 open-vocabulary 96-beam word decoder; it still requires matched complete-decoder end-to-end trials, plus genuinely sealed book tests. v170 byte path remains production-protected. No 80% word top-1, frontier-model capability, or formal theorem closure is claimed.