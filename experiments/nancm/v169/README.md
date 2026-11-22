# NaNCM v169 — Native Symbiotic Signed-Cloud 256-Byte Prediction

**Author:** Justin Howard-Stanley · Independent Researcher · Santa Cruz, California, USA.  
**Date:** 2026-11-22  
**Status:** **Experimental, fully native 256-byte predictor validated on development and previously reused TEST windows. NOT promoted.** Active development model remains **v167**. Protected v158 and v148 checkpoint paths remain recoverable. Keep all experiments on existing `master`, not competing branches.

## Why this round matters

The earlier v168 signed ANN improved **candidate recall** but not strict first-choice word accuracy. v169 tested whether the signed cloud genuinely encodes useful predictive context: 244 gate-active DEV contexts from 16 books were queried with the correct preceding text and a fixed, unrelated-book history. ANN top-32 recovered the target **79/244 with real history**, versus **27/244 with unrelated history**. ANN top-1 was 21 vs 4. This proves a *causal-context advantage for the ANN branch on this sample*, not a top-1 win for the full decoder.

A native v169 proposal rescoring option uses both learned channels: S+/S− geometric ANN proposes 64 whole words; v166's trained lexical model gives an independent whole-vocabulary probability to each. A fixed exponent .05 (chosen for candidate top-5 on TRAIN-book calibration) reweights proposals before they enter the exact normalized `P=(1-beta)P167+beta R` word decoder. Exact v168 parity is retained at `V169_NEURAL_RERANK=0`; exact v167 parity is retained at `V168_ANN_STRENGTH=0`. This pathway **did not improve strict word top-1** and must not replace v167.

The substantial result is **symbiotic 256-byte next prediction**. An auxiliary causal byte expert learns 0–3-byte context probabilities from 100 NaNROUTER TRAIN books, and a separate TRAIN-only byte bridge maps the 27-way native word prediction into a normalized alphabet/case/word-boundary/punctuation distribution over all 256 bytes. Two normalized 256-way distributions then combine by a *past-dependent*, but not target-dependent, gate:

`P169(b|history) = (1-eta) P_ngram(b|history) + eta P_signed_lexical_bridge(b|history)`

with `eta=.5` if the past ends with ASCII letters (we are inside a word), and `eta=.1` otherwise. These two weights were **chosen on DEV** and **frozen before evaluating previously reused TEST**. The byte system predicts actual bytes, including uppercase, punctuation and non-ASCII bytes—not just 27 spelling/EOS symbols.

## Matched 256-byte measurements

| Predictive model | 24-book DEV, 8,015 bytes | 24-book reused TEST, 8,003 bytes |
|---|---:|---:|
| Causal smoothed 256-byte trigram only | 2.844988 BPB | 2.820095 BPB |
| Signed-cloud lexical/ANN bridge only | 2.893143 BPB | 2.726648 BPB |
| **Native symbiotic 256-way mixture** | **2.587376 BPB** | **2.505834 BPB** |

The mixture beats the byte-only expert on **24/24** windows on each split. Within-book paired bootstrap confidence intervals for delta BPB (byte-only minus mixture) are approximately DEV **[0.193, 0.322]** and reused TEST **[0.248, 0.383]**. These intervals do not imply an untouched test, because DEV was used for selection and prior TEST windows were already explored in other research. Full-byte performance here is **new architecture's** score; not an old v167 BPB or historical 27-way spelling score.

The byte model is **fully implemented in C++** in `native_byte169.cpp`, compiled into `bin/libwordfield169.so`. Native API:

- `wf169_byte_open(model_path, geometry_path, cloud_path, native_byte_sidecar_path)`
- `wf169_byte_next(handle, past_bytes, past_length, float out[256])`
- `wf169_byte_close(handle)`

The original `wf148_*` word API is unchanged; no inherited weights or optimizer history were replaced. The binary 239,679-byte TRAIN-only `N169BYT1` sidecar is fingerprint-bound to the inherited model/geometry/cloud, with the original lexical and ANN checkpoints preserved separately. The reference Python implementation remains for reproducible parity audits and portable experimentation.

## Matched 96-beam strict next-word checks (NOT 256-byte BPB)

| Metric | Protected v167 | Native v169 lexical-rescored ANN |
|---|---:|---:|
| DEV 480 top-1/top-5/top-32 | 46 / 94 / 150 | 46 / 93 / 158 |
| DEV 1,394 top-1/top-5/top-32 | 141 / 269 / 465 | 141 / 268 / 480 |
| Reused TEST 480 top-1/top-5/top-32 | 46 / 104 / 162 | 47 / 103 / 165 |

The additional native lexical rerank only modestly improves symbol loss over v168; it does **not** satisfy the word-promotion gate. A separate TRAIN-book modal portfolio reranker improved internal validation 24→29/190 but dropped independent DEV 27→24/244, **rejected**. A document-local two-word copy cache was too sparse, 26 matches and only two correct out of 1,394 words, **rejected**. Direct word-ranking using byte scores fell from native 12→7 top-1 on a 100-context pilot, **rejected**. The 80% first-choice goal is **still open**.

## Build, run and regression

From the **complete v169 recovery ZIP**, not the source-only GitHub folder:

```sh
OPENBLAS_NUM_THREADS=1 make -B v169
OPENBLAS_NUM_THREADS=1 make test-v169
OPENBLAS_NUM_THREADS=1 make test-v169-byte
# Full native + streaming tests:
OPENBLAS_NUM_THREADS=1 make test-v169-full
# Rebuild deterministic TRAIN-only native 256-byte sidecar:
python experiments/nancm/v169/pack_byte169.py
# Example direct 256-way next-byte predictions:
OPENBLAS_NUM_THREADS=1 python experiments/nancm/v169/byte_system169.py
```

For the C++ byte API, configure the frozen v164/v166/v167/v168 environment as in the runner scripts, set `V168_ANN_STRENGTH=.05`, `V169_NEURAL_RERANK=.05`, and pass `checkpoints/v169_native_byte_lattice.bin` to `wf169_byte_open`. It requires the previous trained checkpoints from this ZIP. Use `V169_NEURAL_RERANK=0` for exact v168 word-probability rollback or `V168_ANN_STRENGTH=0` for exact v167 rollback.

Regression: **450 exact v168 rollback vectors; 300 exact v167 vectors; 1,440 normalized word-prefix vectors and 1,440 future-suffix exclusion vectors; 4,545 exact historical documentation vectors; 288 cross-language 256-way distributions with maximum absolute difference 3.6e−8; 8,015 DEV and 8,003 TEST byte probabilities scored directly from C++ and matched against frozen Python references**. Missing/malformed byte sidecars fail closed.

## Scientific and production limitations

1. This is a new 256-way hybrid architecture. The historical native word generator by itself still lacks a 256-byte API.
2. The same 24-book TEST windows have been reused across prior experiments. No pristine independent-book certification or inherited-pretraining contamination audit has been completed.
3. The byte-level gain **does not** imply better strict word top-1; the top-1 target remains ≈10% on these samples, nowhere near 80%.
4. True +2/+3 *without revealing intervening bytes*, subword/BPE, runtime throughput on Android/Termux, large raw-corpus continuation, and downstream transfer remain open.
5. The n-gram byte expert uses a Python-trained sidecar but **inference is now native C++**. No old weights or optimizer moments were altered.

All numerical evidence, source, trained sidecars, optimizer checkpoints, logs, claim ledgers, manifests, and checksums are inside this single cumulative ZIP. Original signed 128+128 cloud and all predecessor architectures preserved.