# NaNCM structural development — sequential checklist

One cumulative checkpoint lineage. Preserve inherited parameters, Adam moments, group steps and RNG; never claim promotion merely from successful workflow exit. Keep GPU full training disabled until final acceptance. Use verified NaNROUTER corpora and wordlists, book-disjoint DEV, separate untouched TEST. Current 360-target and fresh windows are engineering DEV, not final TEST.

## Stage 1 — Geometry (v29, in progress)
- [x] 256 points / all 32,640 pair addresses (v26 verified).
- [x] Independently nonlinear support and opposition (v26 gradient tests).
- [x] Eight-coordinate bidirectional recurrence implemented (v27 corpus run completed; causal control failed and promotion withheld).
- [x] Signed point-local coordinate exchange implemented (v29, standalone CPU preflight passed).
- [x] v29 exact zero-extension prediction and prefix parity on actual v28 checkpoint.
- [x] v29 gradients and exchange-off / wrong-history diagnostic completed; direct effect small.
- [ ] Match held-out full-word NLL, spelling/EOS, certified word top-1/top-5; require loss improvement and no meaningful rank regression before promotion.
- [x] v29 bounded recurrence and channel utilization verified.

## Stage 2 — Temporality (v30 implementation; parent acceptance pending)
- [x] Persistent signed rotating state across observed words; reset at document boundaries implemented in v30.
- [x] Differentiable 96-word closed-form scan and streaming state update agree in isolated CPU preflight (1.39e-17 max error).
- [x] v30 integrated stream law and session restore checks passed (short history).
- [x] Four trainable decay timescales, independent signed phase and amplitude implemented.
- [ ] Extend 96-word bounded differentiable scan to carried-state chunk training and longer held-out horizons.
- [ ] Compare 6/32/128/512+ word horizons; shuffle and reset ablations; suffix invariance.
- [ ] Save session state and recurrent optimizer state; resume equivalence.

## Stage 3 — Higher-order geometry (v31 implementation; corpus pending)
- [x] Signed dynamic edge state during recurrent solver passes implemented; deterministic pair ordering.
- [ ] Persistent edge state across observed words (currently only solver-local).
- [x] Sparse 1,024-triangle incidence on all 256 points; 2,816 edges; 49,152 new parameters; local CPU zero parity/gradient/control preflight passed.
- [ ] Expand or learn active triangle selection, preserve full combinatorial address map.
- [x] Same-parameter pair-operator, triangle-off and edge-reset ablations implemented.
- [ ] Run independently trained matched-budget pair-only control; verify held-out improvement and resource cost.

## Stage 4 — Lexical representation (v32 experimental implementation)
- [x] Tied trainable 16-D lexical codebook with eight causal intake rails and lower/upper excitation; shared output scoring; standalone CPU preflight passed.
- [ ] Integrated checkpoint and corpus gradient/likelihood verification.
- [ ] Train class and within-class competition on deployed normalized law; OOV escape.
- [ ] Hard negative and confusable word discrimination; report exact word rank and NLL.

## Stage 5 — Byte processing (v33 experimental implementation)
- [x] Raw 256-way causal byte head, tied byte codebook and lower/upper signed-field excitation; raw bytes retain all punctuation/case/whitespace.
- [x] Standalone byte geometry CPU preflight: six nonzero gradients, max abs finite difference discrepancy 2.37887e-11, zero residual baseline parity.
- [ ] Integrated corpus and checkpoint gradient verification; compare full, byte-local, input-off, wrong-word-context and count-only laws.
- [x] Positive normalized byte probabilities and rational interval encoder/decoder implemented.
- [ ] Verify real adaptive raw-byte codec and normalized distribution on corpus.
- [ ] Derive consistent cross-level raw-byte/word marginals (separate laws in v33).
- [ ] Report true bits/byte and byte top-1/top-5 on held-out corpus; report word regression separately.

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

- v28 full-word calibration NLL 6.539287 -> 6.187458; field top-1 96/360, top-5 164/360, spelling/EOS 1.733220. Retrieval gate mean .7143. Exact deployed likelihood and prefix factorization passed.
- v29 calibration NLL 6.187458 -> 6.185081, field top-1 94/360, top-5 165/360, spelling/EOS 1.733857; exchange-off delta only .00003414 NLL on 256 calibration targets; promotion withheld.
- v31 local triangle preflight: max finite difference error 3.81e-19, all points covered, zero parity, stateful edge sensitivity, pair control distinct. Corpus pending; no stage acceptance.

- v30 corpus complete: calibration NLL 6.185081 -> 6.183417, field top-1 95/360, top-5 165/360, spelling/EOS 1.732560. Wrong-history vs true-history difference negligible; no promotion.
- v32 local PyTorch preflight: four nonzero gradients, max abs finite difference error 2.06809e-10, exact zero extension and analytic readout gradient parity. Full corpus pending. Tied codebook starts from inherited lexical vectors plus small seeded noise; readout/intake projection weights start zero.

- v33 adds 256 byte embeddings plus masked pad, signed-field intake and tied output; standalone PyTorch CPU test executed, integrated raw corpus pending. Count-only baseline fitted on 96 reference books; byte pilot samples separate 288 TRAIN books; calibration and held-out book IDs retained. No full training.
