# v169 research log — 2026-11-22

1. Recovered complete 166.9 MB v168 signed-ANN archive, verified source/weights, protected v167 baseline and frozen 480/1,394-word results.
2. Tested Q166 neural rescoring of R168 ANN candidates. TRAIN-book modal search suggested small exponents; matched native C++ v169 implemented with `V169_NEURAL_RERANK`, .05 accepted as conservative research. DEV 480 top-1 still 46; all 1,394 top-1 still 141. Not word-promotable.
3. Studied local causal suffix reuse on DEV; ≥2-word context repeated at only 26/1,394, correct just 2, discarded.
4. Trained/validated a larger 3-expert candidate portfolio ranker using 850 TRAIN examples from 91 books, book-separated calibration. TRAIN validation 24→29/190 top-1, but independent DEV active subset 27→24/244; rejected.
5. Paired genuine versus unrelated-book history signed ANN queries on 244 gate-active DEV positions (16 books). True history recovers target top-1 21 vs 4, top-32 79 vs 27. Signed geometry has useful causal information but poor direct final ranking.
6. Constructed fully normalized 256-way word bridge (case/nonletter/word boundary from TRAIN counts) and independent byte ngram (orders 0–3; smoothed), fitted using 100 TRAIN books/300 windows.
7. DEV scores: ngram 2.844988, word bridge 2.893143; normalized mixture with .5 inside-word and .1 outside-word obtains 2.587376 BPB. Weights chosen on DEV then frozen.
8. Reused TEST score after freezing: ngram 2.820095, word bridge 2.726648, mixture 2.505834; positive versus ngram on 24/24 windows. This TEST is not pristine due prior use.
9. Implemented `byte_system169.py` as actual `next256(past)` streaming interface with immutable SHA-256 locks; 288 reference predictions passed, all positive/normalized.
10. Moved byte n-gram, case, punctuation and mixing to native C++ `native_byte169.cpp` with `wf169_byte_open/next/close` and 239,679-byte `N169BYT1` TRAIN-only model-fingerprinted sidecar. 288 complete 256-way parity vectors match Python to max abs difference 3.56e−8; entire DEV 8,015 and reused TEST 8,003 native predictions reproduced identical BPB; invalid sidecars fail closed.
11. Tested using byte-mixture word scores to rerank the native portfolio; first 100 active contexts lost top-1 12→7, rejected.
12. No active word promotion; cumulative archive created, retained inherited optimizers, consolidated source/results/manifest/checksums and GitHub `master` implementation.

Pending: genuinely pristine books, full contamination audit, +2/+3 free-run conditional inference, Android build, more powerful learned long-context first-choice word generator, 80% strict word top-1 objective.