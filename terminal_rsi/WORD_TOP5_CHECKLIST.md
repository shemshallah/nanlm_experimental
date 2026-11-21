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
| W0.4 | Coverage / OOV audit by domain | TODO | 4k vs 16k, raw vs normalized tokenization |
| W0.5 | Punctuation/case-normalized tokenizer | TODO | lower/strip punctuation; rebuild vocab |
| W0.6 | 16k vocabulary coverage | TODO | measure ceiling; flat/PQ later |
| W0.7 | UNK calibration / copy surface | TODO | logistic UNK gate; report observed separately |
| W0.8 | Normalize all priors to log-probabilities | TODO | current 4-gram scatter is unnormalized |
| W0.9 | Book/chunk sentinels for retrieval | TODO | no matches across books |

## 1. Retrieval / compression / memory

| ID | Item | Status | Test / result |
|---|---|---|---|
| W1.1 | Word PPM orders 1-8, longest-match + backoff | TESTED | 2,608 units: top1 31.75%, top5 51.92%, top20 59.97% |
| W1.2 | Exact suffix array / FM-index | TODO | collision-free counts, PPM exclusion |
| W1.3 | CTW / context-tree weighting | TODO | Bayesian depth mixture |
| W1.4 | LZP last-occurrence predictor | TODO | confidence-gated copy |
| W1.5 | Recency / burstiness cache | TODO | document cache, decay |
| W1.6 | Copy / entity / name cache | TODO | capitalized entities, pronouns |
| W1.7 | BM25 / kNN context retrieval | TODO | rare-word inverted index |
| W1.8 | Phrase table / Re-Pair grammar | TODO | 2-6-gram maximal phrases |
| W1.9 | Solomonoff/MDL program library | DEFERRED | after CTW; copy/repeat/grammar programs |
| W1.10 | Multi-granularity word+BPE+byte index | TODO | fallback for OOV/morphology |

## 2. Candidate generation / reranking

| ID | Item | Status | Test / result |
|---|---|---|---|
| W2.1 | Candidate union (head, PPM, beam, cache, bridge) | TODO | recall@32/64/128/256 |
| W2.2 | Listwise reranker (MLP/GRU) | TODO | listwise + top5 hinge |
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
| W3.4 | 16k flat head upper bound | TODO | coverage vs ranking |
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
| W5.1 | Clonal-selection immune memory | TODO | prototype bank, KNN, recency |
| W5.2 | Echo-state reservoir | TODO | 2 timescales, ridge readout |
| W5.3 | HRR / holographic reduced representations | TODO | circular convolution, bundling |
| W5.4 | STDP spiking trace memory | TODO | sparse codes, delays, traces |
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
