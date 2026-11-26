# NaNCM structural development — sequential checklist

One cumulative checkpoint lineage. Preserve inherited parameters, Adam moments, group steps and RNG; never claim promotion merely from successful workflow exit. Keep GPU full training disabled until final acceptance. Use verified NaNROUTER corpora and wordlists, book-disjoint DEV, separate untouched TEST. Current 360-target and fresh windows are engineering DEV, not final TEST.

## Stage 1 — Geometry (v29, in progress)
- [x] 256 points / all 32,640 pair addresses (v26 verified).
- [x] Independently nonlinear support and opposition (v26 gradient tests).
- [x] Eight-coordinate bidirectional recurrence implemented (v27 corpus run completed; causal control failed and promotion withheld).
- [x] Signed point-local coordinate exchange implemented (v29, standalone CPU preflight passed).
- [ ] Verify exact zero-extension prediction and prefix parity on actual v28 checkpoint.
- [ ] Verify every new gradient; compare exchange-on/off and wrong-history controls.
- [ ] Match held-out full-word NLL, spelling/EOS, certified word top-1/top-5; require loss improvement and no meaningful rank regression before promotion.
- [ ] Check bounded recurrence and per-channel utilization.

## Stage 2 — Temporality (v30 implementation; parent acceptance pending)
- [x] Persistent signed rotating state across observed words; reset at document boundaries implemented in v30.
- [x] Differentiable 96-word closed-form scan and streaming state update agree in isolated CPU preflight (1.39e-17 max error).
- [ ] Verify integrated stream/prefix law and session restore on corpus checkpoint.
- [x] Four trainable decay timescales, independent signed phase and amplitude implemented.
- [ ] Extend 96-word bounded differentiable scan to carried-state chunk training and longer held-out horizons.
- [ ] Compare 6/32/128/512+ word horizons; shuffle and reset ablations; suffix invariance.
- [ ] Save session state and recurrent optimizer state; resume equivalence.

## Stage 3 — Higher-order geometry
- [ ] Stateful signed edge dynamics with conserved address mapping.
- [ ] Sparse/factorized triangle interactions over C(256,3)=2,763,520 possible triples.
- [ ] Compare matched parameter/compute pair-only control and triangle ablation; measure active triples and memory.

## Stage 4 — Lexical representation
- [ ] Learned word-to-field and field-to-word embeddings with shared contextual credit.
- [ ] Train class and within-class competition on deployed normalized law; OOV escape.
- [ ] Hard negative and confusable word discrimination; report exact word rank and NLL.

## Stage 5 — Byte processing
- [ ] Raw 256-way causal byte head with byte-to-field feedback, punctuation/case/whitespace preserved.
- [ ] Exact normalization, prefix/word consistency and byte codec roundtrip.
- [ ] Report true bits/byte and byte top-1/top-5, not spelling/EOS proxy.

## Stage 6 — Subword processing
- [ ] Trainable compositional subword encoding, reversible byte segmentation.
- [ ] Field-mediated interaction with lexical and byte representations.
- [ ] Token top-1/top-5 and exact segmentation/word probability tests.

## Stage 7 — Attention
- [ ] Field-native selective resonance with positive support and negative suppression.
- [ ] Causal context selection, normalized trust, matched-compute controls.
- [ ] Counterfactual irrelevant/related context, latency and retrieval coverage.

## Stage 8 — Conceptual memory
- [ ] Persistent attractor state and long-range associative recall.
- [ ] Admission/eviction without future leakage or cross-book state contamination.
- [ ] Correct/unrelated/paraphrased history, long-range query tests.

## Stage 9 — Unified learning
- [ ] Exact deployed likelihood for words/prefixes, raw byte and subword heads.
- [ ] Joint optimization through all enabled pathways, monitor gradient scales and interference.
- [ ] Checkpoint replay/rollback tests and matched multi-level ablations.

## Stage 10 — GPU engineering
- [ ] Vectorized/batched sparse propagation and fused kernels where profiled.
- [ ] CPU/T4/Blackwell settings on same checkpoint schema; dtype and probability parity.
- [ ] Throughput, VRAM, time-to-quality and 10-minute GPU smoke gate; no long training.

## Stage 11 — Final validation
- [ ] Frozen corpus manifest with book IDs, hashes, train/cal/DEV/TEST isolation; cross-domain and untouched TEST.
- [ ] Byte BPB, byte/token/word top-1/top-5, exact word top-5, uncertainty by book.
- [ ] Multi-seed matched budget learning curves, component ablations, throughput/memory.
- [ ] Approve long GPU run only with demonstrated improvement over strongest matched parent/baseline.

## Verified evidence and pending runs
- v24, v25, v26 workflows succeeded. v26 word top-1 68/360, top-5 145/360, spelling/EOS 1.842414 bits/symbol. Known-word calibration NLL 6.459613 -> 6.448943. Rank tradeoff remains; no blanket promotion.
- v27 workflow 45896486886; v28 workflow 45896650014 pending when v29 was authored.
- v29 standalone exchange: float64 finite differences max abs error 3.40e-17; zero extension exact; cross-coordinate activation nonzero; joint theoretical fixed-input contraction bound .905. This is not a corpus benchmark.

- v27 corpus complete: field top-1 63/360, top-5 149/360, calibration NLL 6.448943 -> 6.461300. Shuffled-history control slightly better than actual; no promotion.
- v29 run 45897647647 active; v30 code staged in sequence, dependent on v29 checkpoint. Experimental lineage is not accepted production lineage.
