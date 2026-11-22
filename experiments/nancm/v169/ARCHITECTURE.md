# Native v169 architecture — signed geometry × lexical birth × causal byte statistics

## Word architecture (inherited, frozen)

`128 lower concept points ⇄ 128 upper concept points`, with oriented S+/S− edge tensors and learned 32D lexical token-context vectors (trained v166). The 199,956 TRAIN-address v168 ANN index stores 32D lexical + 16D signed spectral embeddings and up to eight word continuations per address. At inference, 32 nearest addresses produce a 64-word normalized ANN proposal `R168(w|history)`. The v169 research word path multiplies these candidate weights by `Q166(w|history)^0.05`, renormalizes, then mixes with the native v167 whole-word distribution, preserving parent prefix/EOS conditionals. This part did not improve strict word top-1.

## Native 256-byte pathway (new)

`P_byte_ngram`: hierarchical counts for actual bytes 0..255, orders k=0,1,2,3; recursive Dirichlet backoff with strengths 10/30/80 and positive smoothed zero-order prior. Counts exclusively from 100 TRAIN books (300 windows), not DEV/TEST.

`P_word_bridge`: native word predictor supplies 27 symbols for the completed preceding word context and current lowercased partial ASCII word. If inside an ASCII word, its 26 letters provide continuation probabilities and EOS supplies probability of any non-letter; if outside, a TRAIN-only two-byte causal word-start model determines the alphabet/nonalphabet split. A TRAIN-only case model divides each letter between lowercase/uppercase; a TRAIN-only nonalpha n-gram divides the residual among all 204 non-ASCII-letter bytes. All factors are normalized, positive, and target-independent.

`P169(next_byte | actual past) = (1 - eta(past)) P_byte_ngram + eta(past) P_word_bridge`

`eta(past)=0.5` if `past` ends in ASCII letters; `eta(past)=0.1` otherwise. The mode is entirely determined by **past**, never by the correct next byte. Full 256-byte normalized score is computed in `native_byte169.cpp` and exposed by `wf169_byte_next`. Frozen mixture strengths selected on DEV, TEST reused later.

The binary sidecar `N169BYT1` contains model/geometry/cloud fingerprints; 9,601 n-gram rows, 519 byte-start rows, 87 nonletter rows, and four case counts. The native loader rejects wrong fingerprints, malformed counts, duplicate keys, extra bytes and missing files. Inherited S+/S− and AdamW optimizer weights are preserved exactly.

Causality: model receives only `past` bytes and uses only complete previous words plus a partial current word. The `next256` method returns the whole 256-symbol predictive distribution, including punctuation and UTF-8 bytes. Testing compared all 256 probabilities at 288 selected positions against an independent Python reference, then scored every byte in paired DEV and reused TEST windows.

## Caveat

The big BPB gain comes from combining two imperfect experts, **not** from having a higher word top-1. Its proof-of-concept value merits a separate untouched-book preregistered evaluation, and it must not be represented as an 80% word predictor.