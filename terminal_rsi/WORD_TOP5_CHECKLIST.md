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
