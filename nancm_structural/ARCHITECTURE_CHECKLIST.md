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
- [x] v32 integrated gradients and deployed likelihood passed; fresh-window and calibration regression, no promotion.
- [ ] Train class and within-class competition on deployed normalized law; OOV escape.
- [ ] Hard negative and confusable word discrimination; report exact word rank and NLL.

## Stage 5 — Byte processing (v33 experimental implementation)
- [x] Raw 256-way causal byte head, tied byte codebook and lower/upper signed-field excitation; raw bytes retain all punctuation/case/whitespace.
- [x] Standalone byte geometry CPU preflight: six nonzero gradients, max abs finite difference discrepancy 2.37887e-11, zero residual baseline parity.
- [x] v33 integrated gradients and controls passed; full field BPB worse than count-only across all three domains.
- [x] Positive normalized byte probabilities and rational interval encoder/decoder implemented.
- [x] v33 real adaptive codec 16/16 exact roundtrip and positive normalization.
- [ ] Derive consistent cross-level raw-byte/word marginals (separate laws in v33).
- [x] v33 actual raw-byte metrics and matched 360 word regressions reported.

## Stage 6 — Subword processing (v34 experimental implementation)
- [x] Complete prefix tree, 2,296 leaves; tied byte compositions, exact segmentation with explicit pending suffix; local CPU test passed.
- [ ] Learn adaptive prefix-tree topology; expand beyond two-byte leaves.
- [x] Composed token intake into both signed halves; shared byte codebook and field-conditioned token second-byte distribution.
- [ ] Joint raw-word marginal consistency; token input causal controls.
- [x] Implement exact token ranking over all leaves and first-byte marginal checks, byte-level metrics and calibration-only trust.
- [x] v34 corpus and adaptive codec passed; token metadata loaded by v35 implementation.
- [ ] Full checkpoint reload and acceptance; v34 causal effect negligible.

## Stage 7 — Attention (v35 implemented, corpus pending)
- [x] Anchor-conditioned competitive point allocation, signed opposition, bounded recurrent routing and dense signed readout implemented; local CPU preflight passed.
- [x] Same-parameter uniform attention and resonance-off controls; contrastive correct-versus-other-book extended history objective implemented.
- [ ] Independently train uniform routing control at matched budget; compare held-out causal benefit.
- [ ] Corpus results, true-vs-wrong history gap, latency and retrieval coverage; related-history control still required.

## Stage 8 — Conceptual memory (v36 experimental implementation)
- [x] Document-local 1,024-entry bounded associative store; learned lexical, identity and role keys; field-conditioned support/opposition retrieval and normalized sparse word mixture implemented.
- [ ] Trainable admission/eviction, bidirectional memory-to-field recurrence and persistent long-horizon state beyond 1,024 entries.
- [x] Causal observe-after-predict admission, FIFO eviction, session reset/restore implemented; standalone CPU checks passed.
- [ ] Integrated corpus session and checkpoint reload verification.
- [x] Correct vs different-document memory, uniform and exact-match controls, candidate recall at minimum ages 32/128/512, long-window corpus benchmark implemented.
- [ ] Paraphrase controls, independently trained alternatives, acceptance on untouched TEST.

## Stage 9 — Unified learning (v37 experimental implementation)
- [x] Bayesian canonical lowercase word-event/raw-byte bridge with exact event likelihood and raw escape; isolated CPU preflight passed.
- [ ] General raw-word marginal (case, punctuation, Unicode) and bridge-derived token ranking; separate existing token law retained.
- [x] Per-group symmetric gradient projection and transactional Adam proposal with both sampled word/token loss gates; explicit conflict counts and rollback assertions.
- [ ] End-to-end memory gradient through shared field, independently trained control, generalization tests. Memory currently refitted on refreshed cached states.
- [x] v37 calibration-only resonance/input ablations and bridge trust grid; retained inherited moments and frozen word code; failed transactions restore parameters and moments.
- [ ] Integrated corpus and reload/rollback verification.

## Stage 10 — GPU engineering (v38 portable FP32 pilot)
- [x] Portable batched recurrent operator; explicit device placement for inputs and sparse topology; cached device indices.
- [ ] Profile and fuse kernels; persistent device parameter/optimizer storage. Host analytic optimizer still uses NumPy.
- [x] CPU/T4/Blackwell profiles; FP32, TF32 disabled, strict hardware check, shared verified v37 checkpoint and metadata.
- [ ] Run full CPU parity and real T4/Blackwell forward/backward parity on hardware.
- [x] Bounded forward/backward smoke CLI reports throughput, peak CUDA allocated/reserved bytes, gradient/loss/probability parity and unchanged checkpoint; no optimizer update.
- [ ] Execute GPU smoke and time-to-quality benchmarks; no full training until gates.

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

- v31 corpus completed, triangle-off NLL difference approximately 2.8e-7 (no demonstrated benefit). v32 top1 104/360 vs 96/360, fresh BPS and cal NLL regressed. v33 full field raw BPB 2.380073/2.850099/2.606159 vs count 2.348478/2.837475/2.580441; word top1 102/360. No promotion.
- v34 local CPU compositional test: finite difference max 1.15577e-8; 2296 leaves normalized; exact first-byte marginals; 270 bytes reconstructed including incomplete final prefix. Integrated corpus pending. Calibration gate may select zero; do not describe that as learned field improvement.

- v35 local float64 preflight executed: all five groups nonzero, maximum gradient error 1.83536e-13; exact zero route/readout parity. v35 adds 71,680 parameters, frozen inherited phase and joint alternating word/token phase. v34 corpus dependency pending when committed. No promotion.

- v34 actual BPB stories 2.394602 vs count 2.384496; sentences 2.763233 vs 2.766971; paragraphs 2.549439 vs 2.548609. Gate .125/.75; codec 16/16; no promotion.
- v36 standalone local CPU preflight executed: recovered 'answer' after intervening noise, exact state restore, bounded FIFO eviction, empty law, max finite difference error 1.580835462533514e-11. Integrated field-conditioned memory gradient and corpus checks pending. Frozen inherited weights/Adam state during memory pilot; calibration scale includes zero.

- v35 actual corpus: 93/360 top1 vs 102/360 parent; cal NLL 6.355028 vs 6.247382. True vs wrong history disadvantage .122762 nats; resonance-off improves .158653 nats. No promotion.
- v37 local executed CPU gradient surgery: 1000 conflicting random pairs, minimum directional dot 0, exact opposed cancellation. Word-event bridge isolated reference max absolute event discrepancy 1.3877787807814457e-17. v37 uses 192 attempted joint transactions and 128 memory refit steps; only accepted joint steps increment the checkpoint counter. Calibration chooses word ablation, token trust, memory scale, and bridge weight. No GPU/full training.

- v36 long memory candidate48 correct 119/320 vs unrelated 86/320; mean gate .000074487, no word-rank improvement. v38 implements a host/device-autograd bridge for recurrent propagation only, not a full GPU-native trainer. GPU profiles require actual CUDA hardware; no GPU results yet.
