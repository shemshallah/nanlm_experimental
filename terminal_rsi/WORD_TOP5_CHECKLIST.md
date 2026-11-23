# Word Top-5 80% Checklist

Status: TODO / RUNNING / TESTED / INTEGRATED / DEFERRED.
Every TESTED item must record: dev units, top-1, top-5, recall@K, unique hits, cost.
Primary dev: 16 MB pilot, book-disjoint, >=1,000 word units. Secondary: 80 MB.

## 0. Fundamentals and eval integrity

| ID | Item | Status | Test / result |
|---|---|---|---|
| W0.1 | Fix eval double n-gram add | TESTED | fixed; true in-vocab head 12.6%, all-class 42.3% |
| W0.2 | Keep UNK in word n-gram stream | TESTED | fixed; prior can predict UNK |
| W0.3 | 5k+ word-unit eval cache | TODO | 16 books x 256 positions = 1,314 units (partial) |
| W0.4 | Coverage / OOV audit by domain | TESTED | 4k vs 16k, raw vs normalized tokenization |
| W0.5 | Punctuation/case-normalized tokenizer | TESTED | lower/strip punctuation; rebuild vocab |
| W0.6 | 16k vocabulary coverage | TESTED (coverage only) | measure ceiling; flat/PQ later |
| W0.7 | UNK calibration / copy surface | TESTED (UNK channel; no coverage gain) | logistic UNK gate; report observed separately |
| W0.8 | Normalize all priors to log-probabilities | TODO | current 4-gram scatter is unnormalized |
| W0.9 | Book/chunk sentinels for retrieval | TODO | no matches across books |

## 1. Retrieval / compression / memory

| ID | Item | Status | Test / result |
|---|---|---|---|
| W1.1 | Word PPM orders 1-8, longest-match + backoff | TESTED | 2,608 units: top1 31.75%, top5 51.92%, top20 59.97% |
| W1.2 | Exact suffix array / FM-index | TODO | collision-free counts, PPM exclusion |
| W1.3 | CTW / context-tree weighting | TODO | Bayesian depth mixture |
| W1.4 | LZP last-occurrence predictor | TESTED | confidence-gated copy |
| W1.5 | Recency / burstiness cache | TESTED | document cache, decay |
| W1.6 | Copy / entity / name cache | TESTED (copy only) | capitalized entities, pronouns |
| W1.7 | BM25 / kNN context retrieval | TODO | rare-word inverted index |
| W1.8 | Phrase table / Re-Pair grammar | TODO | 2-6-gram maximal phrases |
| W1.9 | Solomonoff/MDL program library | DEFERRED | after CTW; copy/repeat/grammar programs |
| W1.10 | Multi-granularity word+BPE+byte index | TODO | fallback for OOV/morphology |

## 2. Candidate generation / reranking

| ID | Item | Status | Test / result |
|---|---|---|---|
| W2.1 | Candidate union (head, PPM, cache, bridge) | TESTED | recall@32/64/128/256 |
| W2.2 | Listwise reranker (MLP/GRU) | TESTED | listwise + top5 hinge |
| W2.3 | Direct previous-word skip into head | TODO | prev 3 x 64-d embeddings |
| W2.4 | Lexicon-constrained byte beam | TODO | 4k trie, exact word scores |
| W2.5 | Ising/DPP reranker | TODO | PMI fields + diversity |
| W2.6 | Sheaf-theoretic expert fusion | TODO | harmonic extension, H1 fallback |
| W2.7 | Frequency/hubness calibration | TODO | unigram bias, logit adjustment |
| W2.8 | Policy-gradient top-5 tuning | TODO | self-critical, candidate set |
| W2.9 | Word-level mixture of experts | TODO | 8 experts, top-2 routing |

## 3. Extreme classification / label geometry

| ID | Item | Status | Test / result |
|---|---|---|---|
| W3.1 | Brown/bisecting cluster hierarchical softmax | TODO | 64-128 clusters |
| W3.2 | Spherical / PMI-SVD label embeddings | TODO | PPMI-SVD init, cosine head |
| W3.3 | Product-quantized 16k tail | TODO | 2x128 / 3x32 codebooks |
| W3.4 | 16k flat head upper bound | TESTED (coverage only) | coverage vs ranking |
| W3.5 | Low-rank / hyperbolic label geometry | DEFERRED | after spherical |
| W3.6 | Class-based n-gram backoff | TODO | cluster transitions |

## 4. Linguistic / symbolic

| ID | Item | Status | Test / result |
|---|---|---|---|
| W4.1 | POS/supertag prior | TODO | tag stream, P(tag|history), P(word|tag) |
| W4.2 | Construction grammar / typed slots | TODO | `the ADJ of`, `as soon as` |
| W4.3 | Centering / entity grid | TODO | Cb/Cf, pronoun/name bias |
| W4.4 | Morphology / FST factorization | TODO | lemma + inflection + case + punctuation |
| W4.5 | Structure mapping / typed skeletons | TODO | role-aligned analog retrieval |
| W4.6 | WordNet / selectional preferences | DEFERRED | requires WordNet; optional |
| W4.7 | Conceptual blending | DEFERRED | after W4.1-W4.5 |
| W4.8 | CCG parser candidate pruning | DEFERRED | after POS gate |

## 5. Alien computation

| ID | Item | Status | Test / result |
|---|---|---|---|
| W5.1 | Clonal-selection immune memory | TESTED (kNN proxy) | prototype bank, KNN, recency |
| W5.2 | Echo-state reservoir | TODO | 2 timescales, ridge readout |
| W5.3 | HRR / holographic reduced representations | TESTED | circular convolution, bundling |
| W5.4 | STDP spiking trace memory | TESTED (decayed-transition proxy) | sparse codes, delays, traces |
| W5.5 | Kuramoto oscillator prior | TODO | phase advance, resonance |
| W5.6 | Cellular automaton / hypergraph | DEFERRED | after cheaper aliens |
| W5.7 | Ant-colony / Physarum flow | TODO | word graph diffusion |
| W5.8 | Ising/Hopfield associative memory | TODO | candidate spins, free energy |
| W5.9 | Topology / persistent homology gate | TODO | H0 merge-tree features |

## 6. Alien mathematics / information theory

| ID | Item | Status | Test / result |
|---|---|---|---|
| W6.1 | Lexicon-constrained byte beam (same as W2.4) | TODO | byte trie + word trie |
| W6.2 | Sheaf fusion (same as W2.6) | TODO | train/eval consistency |
| W6.3 | CTW + program library (W1.3/W1.9) | TODO | depth mixture |
| W6.4 | Renormalization-group phrase blocks | TODO | Re-Pair block prediction |
| W6.5 | Free-energy switching regimes | TODO | K=64 latent regimes |
| W6.6 | Tensor-train / MPS word model | DEFERRED | after n-gram ceiling |
| W6.7 | Persistent homology (W5.9) | TODO | H0/H1 features |
| W6.8 | Quantum-inspired Born/Grover rerank | DEFERRED | shortlist amplification |
| W6.9 | Solomonoff/MDL program induction | DEFERRED | after CTW |

## 7. Ablation and integration

| ID | Item | Status | Test / result |
|---|---|---|---|
| W7.1 | Unified expert cache `.npz` | TODO | logits/features/targets/prev IDs |
| W7.2 | Offline ablation runner | TODO | all sources, recall, top5, unique hits |
| W7.3 | Per-domain breakdown | TODO | prose/code/math/quote |
| W7.4 | Book-level bootstrap CIs | TODO | paired bootstrap |
| W7.5 | Integrate best 3 into 10-min smoke | TODO | no BPB regression |
| W7.6 | Full training integration | TODO | after smoke |

## Current truth

- 4k word head, fixed eval: **12.6% in-vocab top-5**, 42.3% all-class (UNK hits), ~30% OOV.
- PPM context memory: **51.9% top-5**, 60.0% top-20, 69.2% any-order oracle.
- The 80% target is a coverage + candidate-recall problem before it is a ranking problem.

## Measured results (16 MB pilot, book-disjoint)

### Tokenization / coverage / PPM

| config | vocab | OOV | PPM top-1 | PPM top-5 | PPM top-20 |
|---|---:|---:|---:|---:|---:|
| raw, 4k | 4,095 | 31.9% | 32.9% | **54.75%** | 66.87% |
| normalized, 4k | 4,095 | 21.1% | 24.4% | 45.38% | 58.60% |
| normalized, 16k | 16,383 | 8.2% | 17.9% | 33.98% | 47.12% |

Longest-match raw4k: top-1 31.6%, top-5 51.9%, top-20 62.9%.
KT/CTW raw4k: top-1 34.4%, top-5 53.5%, top-20 65.7%.

Coverage improves with normalization/16k, but top-5 drops on 16 MB because contexts
become sparse. 16k needs more data or a much stronger prior.

### Alien / nonparametric

| mechanism | top-1 | top-5 | top-20 |
|---|---:|---:|---:|
| PPM interpolation | 32.9% | **54.75%** | 66.87% |
| LZP (last-occurrence) | 19.7% | 44.34% | 57.91% |
| STDP-like decayed transitions | 31.8% | 43.35% | 55.17% |
| HRR (permutation + ridge) | 31.9% | 43.58% | 52.92% |
| immune kNN (HRR keys) | 30.3% | 35.07% | 51.58% |
| recency cache | 30.3% | 35.93% | 46.24% |
| copy from excerpt | 11.9% | 30.44% | 45.94% |
| alien union (no PPM/cache/copy) | - | **53.11%** | 61.84% |

### Model head, candidate union, reranker

| item | value |
|---|---:|
| fixed eval, in-vocab word top-5 | 12.56% |
| fixed eval, all-class word top-5 | 42.31% |
| head top-5, all test records | 36.36% |
| PPM top-5, all test records | 37.19% |
| candidate recall@20 | 64.88% |
| candidate recall@100 | **79.75%** |
| oracle top-5 of current sources | 51.03% |
| listwise reranker, conditional top-5 | **77.07%** |
| listwise reranker, overall top-5 | **50.00%** |
| listwise reranker, overall top-1 | 37.60% |

**Conclusion:** the selector is good enough to reach 84% *when the target is in the
candidate set*, but the current union covers only 65% at top-20 and 80% at top-100.
The path to 80% requires candidate recall >=95%, which means the lexicon-constrained
byte beam, OOV/copy channel, and/or 16k tail must be added before more reranker tuning.

### Still open / not yet physically tested

W1.2 FM-index, W1.3 CTW, W1.7 BM25, W1.8 phrase/Re-Pair, W2.4 lexicon byte beam,
W2.5 Ising/DPP, W2.6 sheaf fusion, W2.8 policy gradient, W3.1 hierarchical softmax,
W3.2 spherical/PMI-SVD, W3.3 PQ tail, W4.1-W4.8 linguistic, W5.2 reservoir,
W5.5 Kuramoto, W5.6 CA, W5.7 ant/Physarum, W5.8 Ising memory, W5.9 persistence,
W6.4 RG blocks, W6.5 free-energy regimes, W6.6 MPS, W6.8 quantum, W6.9 Solomonoff.

## Top-100 candidate recall experiment (true, all test records)

| item | value |
|---|---:|
| true recall@20 | 64.88% |
| true recall@100 | **79.75%** |
| reranker overall top-5 (top-100 candidates) | 48.86% |
| reranker conditional top-5 | 61.27% |
| source-sum overall top-5 | 40.39% |
| head top-5, all | 36.36% |
| PPM top-5, all | 37.19% |

The earlier 84.4% / recall@50 94.3% numbers were computed only on the
target-in-union subset (selection bias). The true overall reranker is ~49-50%.

**Conclusion:** the candidate generator, not the selector, is now the hard wall.
To reach 80% overall, candidate coverage must rise from 80% to ~95-100%, while the
conditional selector must stay at ~80%. The next mechanisms must create candidates
the current head/PPM/bridge/cache/copy cannot produce: lexicon-constrained byte
beam, OOV/surface copy, 16k tail, and morphological/retrieval candidates.

Still open: W1.2 FM, W1.3 CTW, W1.7 BM25, W1.8 phrase, W2.4 lexicon beam,
W2.5 Ising/DPP, W2.6 sheaf, W2.8 policy, W3.1-W3.3 hierarchical/PMI/PQ,
W3.4 16k ranking, W4.1-W4.8 linguistic, W5.2 reservoir, W5.5 Kuramoto,
W5.6 CA, W5.7 ant/Physarum, W5.8 Ising memory, W5.9 persistence,
W6.4 RG blocks, W6.5 free energy, W6.6 MPS, W6.8 quantum, W6.9 Solomonoff.

## Coverage curve and two-stage selector

True coverage over all 968 test word positions:

| candidate set | recall |
|---|---:|
| head top-100 | 63.2% |
| prior top-100 | 76.0% |
| union top-20 | 64.9% |
| union top-100 | 79.75% |
| union top-500 | 90.08% |
| union top-1000 | **93.29%** |
| union top-2000 | **100%** |
| prior top-1000 alone | 92.56% |

UNK channel: adding class 0 did **not** raise coverage; the missing targets are rare
in-vocabulary words, not OOV.

Two-stage selector (stage 1 linear listwise over head/bridge/prior top-1000 +
PPM/cache/copy; stage 2 MLP on top-50):

| item | value |
|---|---:|
| candidate coverage@1000 | 93.29% |
| stage-1 shortlist recall@50 | 74.97% |
| stage-2 overall top-5 (903 covered) | 52.71% |
| stage-2 overall top-5 over all 968 | ~49.2% |
| stage-2 conditional top-5 | 70.31% |
| stage-2 top-1 | 38.21% |

**New wall:** candidate coverage is sufficient (93% at 1000; 100% at 2000), but the
shortlist/selector loses too much. The 80% path now requires:
1. stage-1 shortlist recall >=95% (currently 75%), and
2. stage-2 conditional top-5 >=85% (currently 70%).

The strongest single source is the prior at top-1000 (92.56% coverage). A source-aware
sheaf/Ising gate over a 1000-candidate prior/head/bridge union, plus more training
data, is the next selector experiment. The 16 MB data budget remains a ceiling.

## MLP stage-1 selector (rerun after server restart)

| item | value |
|---|---:|
| candidate coverage@1000 | 93.29% |
| stage-1 MLP shortlist recall@50 | 75.97% |
| stage-2 overall top-5 (903 covered) | 53.38% |
| stage-2 overall top-5 over all 968 | ~49.8% |
| stage-2 conditional top-5 | 70.26% |

The MLP stage 1 did not materially beat the linear stage 1 (75.0 -> 76.0).
The selector is data/feature-limited, not capacity-limited. With 16 MB and 968
training boundaries, a 1,000-candidate listwise selector cannot reach the ~86%
conditional accuracy needed for 80%.

Next decisive run: 80 MB pilot + 5-6k word-boundary cache + the same selector.
Server restart preserved the 16 MB cache and results.

## Wave 2 modules landed (unit-tested, physical probes pending)

| module | tests | physical result |
|---|---|---|
| `word_lexicon_beam.py` | 36 tests OK (2 skipped) | pending; CLI requires checkpoint+dev-bin+vocab JSON |
| `word_sheaf.py` | 9 tests OK | pending; CLI reads `/marimo/word_dev_cache.npz` |
| `word_linguistic.py` | 36 tests OK | pending; needs dev word-stream probe |
| `word_alien_more.py` | 35 tests OK | probe launched; server died before results |

`word_alien_more.py` includes persistent homology, full STDP spiking trace,
reservoir, Kuramoto, cellular automaton, ant colony, tensor-train/MPS, Re-Pair
renormalization blocks, and free-energy regimes.

The MLP stage-1 selector result is recorded above (53.38% overall on 903 covered,
70.26% conditional); it did not break the 16 MB selector wall.

Server status: HTTP 410 after the restart; need a fresh Molab pair prompt to run
the sheaf/lexicon/linguistic/alien probes and the 80 MB selector.

## 80 MB run (6-hour checkpoint, 2,240 word boundaries)

Candidate coverage over 1,120 held-out boundaries:

| K | recall |
|---:|---:|
| 20 | 65.7% |
| 50 | 69.0% |
| 100 | 77.7% |
| 200 | 84.2% |
| 500 | 91.0% |
| 1000 | **98.5%** |
| 2000 | 100% |

Single-stage selector sweep (train 1,120 / test 1,120):

| K | coverage | conditional top-5 | overall top-5 |
|---:|---:|---:|---:|
| 50 | 66.8% | **84.5%** | **56.4%** |
| 100 | 77.7% | 69.9% | 54.3% |
| 200 | 84.2% | 64.9% | 54.6% |
| 500 | 91.0% | 56.7% | 51.6% |
| 1000 | 98.5% | 54.5% | 53.7% |

Sheaf fusion CLI (280 fit records): coverage 77.8%, conditional 62.6%, overall 48.7%.

**Conclusion (80 MB):** coverage is solved at top-1000 (98.5%), but the selector's
conditional accuracy and the shortlist coverage trade off almost perfectly:
K50 has 84.5% conditional but only 66.8% coverage; K1000 has 98.5% coverage but
54.5% conditional. Best overall remains 56.4% at K50. The 80% target now requires
a better **shortlist** (high recall at 50-100 candidates) or a much stronger
word-level context model; more candidates alone will not do it.

## CORRECTION: 80 MB cache label-alignment bug

The first 80 MB cache used global training labels with local dev positions.
That invalidated the earlier 80 MB selector numbers (K50 56.4, K1000 53.7).
The cache is fixed: local excerpt word IDs, local previous words, local targets.

Corrected cache: 2,549 records, 64 books.

Corrected coverage (union with GRU source):

| K | recall |
|---:|---:|
| 50 | 84.3% |
| 100 | 89.1% |
| 200 | 90.9% |
| 500 | 94.2% |
| 1000 | 97.6% |

Word GRU:
- small 128/256, 12 epochs: **57.14% top-5** (7,651 dev tokens)
- big 256/512, 20 epochs: **59.22% top-5**

Selector sweep with GRU source (train 1,259 / test 1,290):

| K | coverage | conditional | overall |
|---:|---:|---:|---:|
| 50 | 84.3% | **69.7%** | **58.8%** |
| 100 | 89.1% | 65.7% | 58.6% |
| 200 | 90.9% | 64.1% | 58.3% |
| 500 | 94.2% | 62.1% | 58.5% |
| 1000 | 97.6% | 60.0% | 58.6% |

Best overall remains **58.8%**; the GRU alone (59.2%) is essentially the whole system.
The selector does not add value over the GRU at this data scale.

**Conclusion:** after fixing the cache, the 80% gap is no longer a candidate
coverage problem (97.6% at K1000) or a selector-coverage problem (84.3% at K50).
It is a **word-model capacity/data** problem: the GRU is the strongest predictor
and the selector cannot exceed it materially. The next lever is a larger word
model trained on much more than 80 MB, or a 16k/OOV-aware word model.

## 4-hour sequence-loss continuation (partial, platform kill)

| item | value |
|---|---:|
| resumed from | step 2400 |
| last step | 4080 |
| updates into run | 1,680 |
| elapsed | 9,571 / 14,400 s |
| train BPB | **1.96-2.07** |
| bit accuracy | 92.0% |
| rate | 0.18 updates/s |
| last Kaggle version | 06:52 UTC, step ~4050 + log |
| final metrics | **not written**; container killed at 06:52 UTC |

The all-position loss reduced train BPB from ~2.2 to **1.96** by step 4080.
Remaining budget: ~4,800 s plus final DEV/WikiText eval. Resume from the 06:52
Kaggle checkpoint on the next session.

## FINAL: 4h continuation finish (step 4170)

| metric | value |
|---|---:|
| final step | 4170 |
| updates in finish | 220 |
| DEV BPB (256 ctx) | **1.9561** |
| DEV byte top-1 | **60.94%** |
| DEV byte top-5 | **89.06%** |
| DEV token beam top-5 | 87.92% |
| DEV word beam top-5 | 46.67% |
| word head top-5 | 54.98% |
| word_observed top-5 | 27.17% |
| unit decoder delta | +0.0469 BPB |
| WikiText-2/103 test BPB | **2.6342** |
| WikiText test byte top-5 | **80.66%** |
| WikiText test byte top-1 | 51.17% |
| WikiText-103 valid BPB | **2.5578** |
| WikiText valid byte top-5 | **79.49%** |
| WikiText valid byte top-1 | 52.54% |

Kaggle final version: `final.pt` + `metrics.json` + full train log at 17:35 UTC.
Final metrics dataset: `shem42/trsi-100m-final-metrics`.

This is the best model so far: DEV below 2 BPB, WikiText test 2.63 BPB / 80.7% byte top-5.
The word head remains the weak component (55% all-class, 27% observed).

## Sequence-loss curve and training-time extrapolation

Observed (batch BPB, noisy): 2.299 at step 2400 -> ~2.04-2.07 at 3900-4170.
Rate: 0.183 updates/s = **657 updates/hour**.
20-minute finish: 220 updates, final DEV BPB 1.956, DEV byte top-5 89.06%.

Fit from the sequence-loss segment (train-batch BPB, RMSE ~0.020):

| additional time | updates | total step | optimistic BPB | conservative BPB |
|---:|---:|---:|---:|---:|
| 20 min | 219 | 4,389 | 1.989 | 1.996 |
| 1 h | 657 | 4,827 | 1.931 | 1.944 |
| 2 h | 1,315 | 5,484 | 1.848 | 1.875 |
| 4 h | 2,629 | 6,799 | 1.701 | 1.762 |
| 6 h | 3,943 | 8,113 | 1.574 | 1.676 |
| 12 h | 7,887 | 12,057 | 1.292 | 1.522 |
| 24 h | 15,774 | 19,944 | 0.998 | 1.424 |

Optimistic asymptote 0.8, tau 8,685; conservative asymptote 1.4, tau 4,836.
These are train-batch trends, not guarantees; the last 20 minutes showed little
batch-BPB movement despite a better DEV checkpoint.

Throughput/cost:
- 657 updates/hour; 50-step save every ~4.6 min; checkpoint 1.39 GB.
- 10-minute Kaggle versioning = 6 versions/hour, ~8.3 GB/hour retained.
- 4 h = 2,629 updates; 6 h = 3,943; 12 h = 7,887; 24 h = 15,774.
- Add ~25 min staging per fresh/resumed session; Molab sessions last ~2.5-3.5 h,
  so 4 h+ needs rolling Kaggle resume.

Naive offset projection (current train 2.07 -> DEV 1.956, Wiki test 2.634):
4 h -> DEV ~1.59-1.65, Wiki ~2.26-2.33; 6 h -> DEV ~1.46-1.57, Wiki ~2.13-2.24.
Treat these as rough, not measured.

## Second 20-minute run: flattening check

| item | step 4170 | step 4392 | delta |
|---|---:|---:|---:|
| DEV BPB (256 ctx) | 1.9561 | **1.9311** | -0.0250 |
| DEV byte top-1 | 60.94% | 58.59% | -2.35 pp |
| DEV byte top-5 | 89.06% | 88.28% | -0.78 pp |
| train batch BPB | ~2.06 | ~2.00 | ~flat |
| unit decoder delta | +0.0469 | +0.0491 | +0.0022 |

222 updates in 1,200 s (0.184 updates/s). The batch-BPB trend has flattened
around 2.0; DEV BPB improved only 0.025 and top-5 moved down within the noise of
a 256-position eval. Another 20-minute increment is not expected to move the
headline metrics materially.

## Arbitrary vocabulary size + 16k/OOV coverage

Implemented end to end:
- `--word-vocab-size` on `train_verbatim`; default 4096.
- `UnitHeads`, `TerminalRSI`, `CausalHierarchy`, `VerbatimV103RSI`,
  `WordTokenBridge`, `unit_parameter_count`, `hierarchy_parameter_count`,
  `static_upgrade_parameter_count`, and `build_unit_labels` are parameterized.
- Checkpoint top-level `word_vocab_size`; old 4k checkpoints default to 4096.
- 16k run checkpoint: `word_vocab_size=16384`, schema unchanged.
- 43 targeted tests pass, including an 8,192-word head test.

Coverage on the same book-disjoint DEV:

| vocab | 16 MB coverage | 16 MB OOV | 80 MB coverage | 80 MB OOV |
|---:|---:|---:|---:|---:|
| 4k | 68.1% | 31.9% | 71.4% | 28.6% |
| 16k | **82.6%** | **17.4%** | **84.5%** | **15.5%** |

16k smoke (10 min, fresh random, 16 MB pilot, 81 updates):
- DEV BPB 3.508 (not quality; random short run)
- byte top-5 71.1%
- word top-5 25.8% all-class / 20.2% observed
- 16,383 word classes, UNK id 0; 16k bigram table ~1 GB; rate 0.14 upd/s.

16k is not full OOV coverage: 15.5% of 80 MB dev tokens are still outside the
vocabulary. Reaching >95% needs a larger tail (64k/PQ) or a character/byte OOV
fallback, not 16k alone.

## OOV byte fallback + feedback-loop traceroute

Implemented:
- `terminal_rsi/oov_fallback.py`: beam speller over the hierarchy's 256-way
  `unit_byte_raw` decoder; returns OOV surface candidates from a history ending
  at a word start. No target bytes enter the context. 46 tests pass.
- `terminal_rsi/feedback_audit.py`: audits/reopens every hierarchy, MoE and
  symbiosis scalar gate. CLI works on a checkpoint.
- Feedback state now updates even when parent weights are frozen
  (`observe_online_bits` + `observe_verified_byte` unconditionally).
- `fast_adapter` is no longer frozen and is in the `unit_side_heads` optimizer
  group.
- `--reopen-gates` on `train_verbatim` and the resume launcher.

Audit of the step-4170 checkpoint found these disconnected paths:
- `upgrades.symbiosis_gate` (-0.0023)
- `upgrades.backbone.1/4/7/10.moe.gate`
- `upgrades.hierarchy.top_moe.gate`

Other hierarchy gates were small but above threshold. A 5-minute continuation
from step 4170 with all of them reopened at 0.1 is running to measure whether
reconnection helps or destabilizes the model.

### OOV speller correction

The first real-model demo used `unit_byte_raw` and produced nonsense
("integrally/integered" after "The quick brown "). The speller now uses the
main 256-way `upgrades["logits"]` byte head. Real checkpoint demo after
"The quick brown " gives "of", "to", "integration(s)" — still a weak LM, but a
real distribution instead of a side-decoder artifact.

Feedback traceroute fixes:
- feedback state updates even when parent weights are frozen
- `fast_adapter` unfrozen and moved into `unit_side_heads`
- `--reopen-gates` for hierarchy/MoE/symbiosis paths
- checkpoint gate audit found dead: symbiosis, four backbone MoE gates, top_moe
- 5-minute gate-reopen smoke (0.1) is running against step 4170.

### Feedback/OOV follow-up

- OOV speller now uses the main 256-way byte head; real checkpoint demo after
  "The quick brown " gives "of", "to", "integration(s)".
- Fast adapter is trained on fresh runs; on legacy resume it is frozen because
  the old four-group Adam state cannot accept extra parameters. `--train-fast-adapter`
  exists but is only safe for checkpoints whose optimizer already has them.
- Gate audit found dead symbiosis, four backbone MoE gates, and top_moe.
  A 5-minute step-4170 continuation with all gates reopened at 0.1 is running
  (`trsi_100m_gate_smoke2`); result pending.
- 46 tests pass.

## Gate-reopen smoke (step 4170 -> 4224)

54 updates, 300 s, all hierarchy/MoE/symbiosis gates reopened at 0.1.

| metric | step 4170 | step 4224 (gates 0.1) |
|---|---:|---:|
| DEV BPB (256 ctx) | 1.9561 | 1.9658 |
| DEV byte top-1 | 60.94% | 58.20% |
| DEV byte top-5 | 89.06% | 88.28% |
| train batch BPB | ~2.0 | ~2.0-2.2 |

The gates did not collapse: after 54 updates every audited gate is still in
0.057-0.10, and `feedback_audit` reports `all gates open`. The 0.01 BPB and
0.8-point top-5 differences are inside the 512-position DEV noise and are not
evidence of a gain. The useful result is that reopening is safe; a fresh run
should start with open gates so the specialist branches receive gradient from
step zero instead of remaining permanently disconnected.

Checkpoint/metrics: `trsi_100m_gate_smoke2`, uploaded at 21:03 UTC to
`shem42/trsi-100m-6h-diverse-checkpoints`.

## Fast-adapter dead loop (fixed)

Traceroute found a second, deeper reason the adapter stayed zero: `RankAdapter`
initialized **both** `A` and `B` to zero. In `(state @ A) @ B`, zero `A` kills
`dL/dB`, and zero `B` kills `dL/dA`. Unfreezing alone could never revive it.

Fixes:
- LoRA-style init: `A ~ N(0, 0.02)`, `B = 0`. Output is still exactly zero at
  init, but `B` receives a nonzero gradient on the first step.
- `reopen_fast_adapter`: legacy all-zero checkpoints get a seeded `A` while
  `B`/`bias` keep their checkpoint values.
- `load_optimizer_allow_extra_params`: legacy four-group Adam state loads by
  removing exactly the new adapter parameters from the side group, then
  restoring them. No inherited moments are lost.
- `--train-fast-adapter` now works on legacy checkpoints; fresh runs always
  train the adapter.
- Real step-4224 checkpoint verification: side group 25 -> 28 params,
  `reopen_fast_adapter=True`, `B.grad.max=14.48`, adapter output 0.0.
- 53 tests pass (7 new adapter/migration/vocab tests).

## 16k warm start from the trained 4k checkpoint

`build_unit_labels` assigns IDs by frequency, so the first 4,096 IDs of the
16k vocabulary are the same words as the 4k vocabulary (same 80 MB corpus).
`expand_word_vocab_state` now grows a trained checkpoint:

- copies the overlapping rows of both word heads, the word bridge embedding,
  bridge output weight, and bridge bias;
- randomizes only the new tail of trained heads; keeps the new bridge output
  rows at zero;
- reinitializes the vocabulary-width internal layers (`unit_to_byte.*`,
  `fast_adapter.A`, `symbiotic_feedback.0.weight`) because their flat feature
  vectors need offset-aware block insertion;
- inherits every non-vocabulary tensor, including the byte backbone, hierarchy,
  MoE, pair memory and gates.

Verified on the real step-4170 checkpoint: 216 tensors expanded, 10
vocabulary-width tensors reinitialized, all 4,096 inherited rows bit-equal,
word heads 4096 -> 16384, model 135.0M parameters.
Enable with `--init-weights <4k.pt> --word-vocab-size 16384
--expand-init-vocab`.

Next fresh run: 16k vocabulary warm-started from `trsi_100m_4h_finish`,
gates opened at 0.1, fast adapter live, word n-gram weight 0.5, OOV byte
speller available as the UNK surface fallback.

## Deeper architecture wiring

The main `train_verbatim` loop was only training `output[0]` (next byte). Two
declared components were computed on every step but never received a loss:

- `output[1]` / `output[2]` (+2 and +3 byte heads),
- `critic` (predicted next-byte surprise, used inside the recursive pass).

New flags:
- `--lookahead-loss-weight`: CE on the +2/+3 heads using targets already inside
  the causal window; the final 1-2 positions are masked by construction.
- `--critic-loss-weight`: MSE between `critic` and detached observed next-byte
  CE, so the recursive feedback signal is a trained surprise estimate instead
  of a random projection.

Both default to 0.0 for old runs. `train_one_batch` activates `upgrade_out`
when either loss is requested, and reports `lookahead_bpb` / `critic_mse`.
`generate_words` now spells UNK word IDs with the byte OOV speller instead of
stopping. 56 tests pass.

20-minute smoke: `trsi_100m_deep_smoke`, resumed from step 4224, gates 0.1,
`--train-fast-adapter`, lookahead 0.1, critic 0.03, word n-gram 0.5.

## 20-minute deeper-wiring smoke (step 4224 -> 4442)

Run: `trsi_100m_deep_smoke`, 218 updates, resumed from step 4224 with
gates at 0.1, `--train-fast-adapter`, lookahead 0.1, critic 0.03,
word n-gram 0.5. Kaggle version 57 (22:07 UTC) has `last.pt` + `metrics.json`
for step 4442.

DEV (512 positions):

| metric | step 4170 | step 4224 | step 4442 (deep) |
|---|---:|---:|---:|
| DEV BPB | 1.9561 | 1.9658 | 1.9963 |
| DEV byte top-1 | 60.94% | 58.20% | 58.59% |
| DEV byte top-5 | 89.06% | 88.28% | 89.84% |
| fast-adapter delta | 0.0 | 0.0 | **+0.0027 BPB** |

Held-out WikiText, same 8 excerpts x 64 positions (paired):

| checkpoint | wt_test BPB | wt_test byte/word top-5 | wt103_valid BPB | wt103_valid byte/word top-5 |
|---|---:|---:|---:|---:|
| step 4170 | 2.6342 | 80.66 / 34.82 | 2.5578 | 79.49 / 28.30 |
| step 4224 gates | 2.6047 | 79.49 / 31.25 | 2.4745 | 79.10 / 29.25 |
| step 4442 deep | 2.5865 | 80.08 / 33.93 | 2.5483 | 78.12 / 28.30 |

The deep run improves WikiText-2 test BPB and word top-5 over the gate
checkpoint (+2.7 points word top-5), but regresses WikiText-103 valid. The
direction is inconsistent across the two held-out sets and the 512-position
samples are noisy, so this is **not** evidence that the lookahead/critic weights
are ready for a long run. They remain opt-in defaults (0.0). The trained fast
adapter is the one unambiguous gain: its held-out contribution is now nonzero
and positive instead of exactly 0.0.

Next tuning run should use a much smaller lookahead weight (0.01-0.02) or
critic-only, and a larger held-out eval before adopting either loss.

## Drastic upgrade: causal copy head in the 100M branch

The v103 parent consumes six causal features per target (BPE/word bigram
hashes, 5-byte match byte/length, 12-byte match byte/length). The 100M
upgrade branch ignored all six. It now has a learned **causal copy head**:

- `upgrades.copy_gate` has one trust scalar for the 5-byte match and one for
  the 12-byte match.
- At the target position the matched candidate byte receives a positive logit
  proportional to `tanh(gate) * match_length / span`, added to the final
  bit logits outside the residual gate.
- Zero-initialized, so legacy checkpoints retain exact parity until
  `--reopen-gates` opens it. `feedback_audit` now audits/reopens it with the
  other feedback gates.
- Legacy optimizer migration handles both the new copy gate in the recursive
  upgrades group and the fast adapter in the side group.

Small smoke on a rebuilt 16 MB diverse corpus (55 updates, warm start from
step 4442, gates 0.1, copy head open):

| metric | value |
|---|---:|
| DEV BPB | 1.8939 |
| DEV byte top-1 | 60.55% |
| DEV byte top-5 | 87.89% |
| learned copy gates | 0.0666 / 0.0670 |
| copy on/off BPB (same DEV) | 2.1240 / 2.1389, **+0.0149 BPB** |

The copy head is a real held-out gain, not just a gate that moved.

The old session's 80 MB pilot was not present on the replacement Molab
session, so a new 80 MB corpus was rebuilt from public HuggingFace sources
(Gutenberg, OpenR1 math, AG News, TinyStories) and a new 80 MB pilot was
generated. The 2-hour run is `trsi_100m_copy_2h`, warm-started from step 4442
with `--reopen-gates 0.1`, copy head active, fast adapter trainable,
lookahead 0.02 and critic 0.01. Rolling backups go to
`shem42/trsi-100m-copy-2h-checkpoints`.

## 2-hour copy-head run (interrupted at 87%)

`trsi_100m_copy_2h` was warm-started from step 4442 on the rebuilt 80 MB
corpus with the causal copy head open (15 gates), fast adapter trainable,
lookahead 0.02 and critic 0.01. It reached **step 1143 at 6243 s / 7200 s**
with train BPB ~1.69-1.96 and bit accuracy ~93%. The Molab kernel then
returned HTTP 410 (session gone) before the final DEV/WikiText evaluation.

Preserved in `shem42/trsi-100m-copy-2h-checkpoints` version 10:
`last.pt` (1.39 GB), `train.log`, `launch.json`. No `metrics.json` or
`final.pt` because the kernel died before the final eval.

Resume needs a fresh Molab URL/token. Resume from the Kaggle `last.pt` with
the same 80 MB pilot (`trsi_80m_deep_pilot`) and `--resume`, then run the
remaining ~950 s plus final evaluation. Do not restart from step 0.

## FINAL: 2-hour copy-head run (resumed and completed)

Resumed exactly from the Kaggle step-1100 checkpoint after the first kernel
died at 6243 s. Rebuilt pilot hashes matched the checkpoint
(`train 6d987e…`, `dev 1f431b…`); the second leg ran 203 updates to step 1303.

DEV on the rebuilt 80 MB corpus: **1.8683 BPB / 58.20% top-1 / 88.28% top-5**.
Learned copy gates: **0.3512 / 0.4067**. Final checkpoint + metrics:
`shem42/trsi-100m-copy-2h-checkpoints` version 12;
paired evaluation JSONs: `shem42/trsi-100m-copy-2h-metrics`.

Paired WikiText, same 8 excerpts x 64 positions:

| metric | deep 4442 | copy 1303 | delta |
|---|---:|---:|---:|
| wt_test byte top-1 / top-5 | 52.15 / 80.08 | **53.32 / 80.86** | +1.17 / +0.78 |
| wt_test token top-1 / top-5 | 36.70 / 77.68 | **37.61 / 79.82** | +0.91 / +2.14 |
| wt_test word top-1 / top-5 | 5.36 / 33.93 | **10.71 / 30.36** | +5.35 / -3.57 |
| wt_test word exact top-5 | 36.61 | **38.39** | +1.78 |
| wt103_valid byte top-1 / top-5 | 52.73 / 78.12 | **53.71 / 80.66** | +0.98 / +2.54 |
| wt103_valid token top-1 / top-5 | 36.19 / 78.73 | **38.41 / 79.37** | +2.22 / +0.64 |
| wt103_valid word top-1 / top-5 | 10.38 / 28.30 | 8.49 / 28.30 | -1.89 / 0.00 |
| wt103_valid word exact top-5 | 27.36 | **30.19** | +2.83 |
| wt_test BPB | 2.5865 | **2.5847** | -0.0018 |
| wt103_valid BPB | 2.5483 | **2.4774** | -0.0709 |

Byte and token top-1/top-5 improved on both held-out sets, and word **exact**
top-5 improved. The word *beam* top-5 is the remaining loss: the byte beam
does not enumerate whole-word candidates well. The next real word lever is a
16k vocabulary plus a lexicon/word-level beam, not more byte tuning.

### Session lifetime

The first 2-hour kernel died at 6243 s because the host keepalive process was
killed (Android), after which Molab idled the kernel out. It was not the 12-hour
platform cap. The replacement session resumed from the rolling checkpoint and
finished the run. Keepalive now runs with `termux-wake-lock` and monitors the
active run; long work should still use 60-90 minute legs plus exact resume.

## 16k 6h run (partial, sandbox hard cap)

Warm-started the 16k model from the 2h copy checkpoint
(`--expand-init-vocab`, 16383 words, copy gates preserved). The run reached
**step 750 / 4200 s of 21600 s** with train BPB ~1.60-1.75 and bit accuracy
~94.2%. The rolling checkpoint is Kaggle
`shem42/trsi-100m-16k-6h-checkpoints` version 4 (step 750).

The Molab kernel then returned **HTTP 410 after ~2h47 from session creation**,
while both the keepalive and the new watchdog were alive and pinging. This is
a hard sandbox lifespan, not an idle timeout; the 12-hour figure in the FAQ is
not the usable limit of this sandbox.

Exact resume kit: `shem42/trsi-16k-6h-resume-kit` (80 MB pilot, corpus, BPE,
launcher). Resume from step 750 with the 80 MB pilot and the same run directory;
remaining training budget ~17400 s. The Kaggle GPU fallback test kernel stayed
queued for >13 minutes, so Molab is the faster path once a fresh pair prompt is
available.

## 16k 60-minute leg (final for this session)

Resumed exactly from step 750 and ran 661 updates to **step 1411** (60 min).

| metric | value |
|---|---:|
| DEV BPB | **1.8346** (from 1.8674) |
| DEV byte top-1 | **62.11%** |
| DEV byte top-5 | **87.50%** |
| fast-adapter delta | **+0.0044 BPB** |
| copy gates | 0.3354 / 0.5212 |
| word-prior gate | 0.0839 |

Saved as `shem42/trsi-100m-16k-6h-checkpoints` version 5, including
`last.pt` (step 1411), `metrics.json`, `train.log`, and `generations.jsonl`.

Generation samples at step 750 -> 1411:

- `"The "` -> `son of the chair has ... done a tongue.` (750)
- `"The "` -> `protection changes in ... Mary story her face ...` (900)
- `"The "` -> `proposal inch of her ... accidents by the world.` (1050)
- `"The "` -> `said, "I have been sorry to ... dear things while I've seen.` (1411)
- `"The quick brown "` -> `when ... KINGO. Here will be to die to the beast ...` (1411)

The model is still far from clean prose, but the 16k continuation is
improving held-out BPB and top-1 while keeping top-5 stable. Session stopped
after the requested 60 minutes; no further training is running.

## 16k second 60-minute leg (step 1411 -> 2060)

Resumed exactly from step 1411 and ran 649 updates in 3603 s.

| metric | step 1411 | step 2060 |
|---|---:|---:|
| DEV BPB | 1.8346 | **1.7974** |
| copy head on / off | — | **1.7974 / 1.8002** (+0.0029 BPB) |
| copy gates | 0.335 / 0.521 | 0.336 / 0.530 |
| fast-adapter delta | +0.0044 | (not re-measured after the failed receipt) |

Checkpoint: `shem42/trsi-100m-16k-6h-checkpoints` version 9 (`last.pt` step
2060). Corrected metrics and generation samples:
`shem42/trsi-100m-16k-metrics`.

The final 512-position top-1/top-5 receipt was lost because the trainer
refused to overwrite the previous leg's `metrics.json`; the 8-probe top-k was
87.5% / 100%. Fixed in `9d2bdcf` (metrics writes are now atomic replacements).
The kernel hit the ~3-hour sandbox cap immediately after the final eval.

Final samples at step 2060:

- `"The "` -> `monstrous of a moment, and ... share a came out of the bird of the water an`
- `"In the beginning, "` -> `all too the story, while ... he had gained up for him, told and the person`
- `"The quick brown "` -> `so was to be be disolated ... to say. ... "We've to put on it with the river`

## 16k third 60-minute leg (step 1961 -> 2595)

The Kaggle dataset's step-2060 "manual" checkpoint was never uploaded; the
actual preserved checkpoint at that point was **step 1961**. The next hour
therefore resumed from 1961 and finished at **step 2595** (634 updates).

| metric | value |
|---|---:|
| DEV BPB | **1.8333** |
| DEV byte top-1 | **62.11%** |
| DEV byte top-5 | **86.72%** |
| fast-adapter delta | **+0.0080 BPB** |
| copy gates | 0.327 / 0.517 |
| word-prior gate | 0.173 |

Checkpoint: `shem42/trsi-100m-16k-6h-checkpoints` version 15
(`last.pt`, step 2595). Metrics/generations:
`shem42/trsi-100m-16k-metrics` version 2.

Final samples at step 2595:

- `"The "` -> `way and shakes the lost ... time over on on the season. How is no more while the`
- `"In the beginning, "` -> `and the little life of the ... children's words was discerned to remain to`
- `"The quick brown "` -> `team, and again. The ... part was to the sovereign of the covenant where the song`

The earlier 1.7974 BPB number came from step-2060 weights that died with the
Molab kernel; it is not a preserved checkpoint. Use step 2595 as the current
16k baseline.

## 16k fourth 60-minute leg (step 2595 -> 3234)

| metric | step 2595 | step 3234 |
|---|---:|---:|
| DEV BPB | 1.8333 | **1.7956** |
| DEV byte top-1 | 62.11% | 59.38% |
| DEV byte top-5 | 86.72% | **87.89%** |
| fast-adapter delta | +0.0080 | **-0.0116** (harmful) |
| copy gates | 0.327 / 0.517 | 0.312 / 0.513 |

639 updates. Checkpoint: `shem42/trsi-100m-16k-6h-checkpoints` version 26
(`last.pt`, `metrics.json`, step 3234). Metrics/generations:
`shem42/trsi-100m-16k-metrics` version 3.

The fast adapter has crossed into slightly harmful territory on this held-out
DEV; the next leg should either freeze it or lower its LR. Top-5 improved while
top-1 and BPB trade off.

Final samples at step 3234:

- `"The "` -> `faces held her of the ... books's hands." ... The face stumped from the flow`
- `"In the beginning, "` -> `and that you know how you must have fancyard of ... the sea of that the door of t`
- `"The quick brown "` -> `at the sergeant. She was shiny ... and the later of the left, and the lette`
