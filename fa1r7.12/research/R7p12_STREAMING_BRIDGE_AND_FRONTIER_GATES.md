# PAL R7.12 — causal streaming, LM bridge, reliability controls, and frontier gate

**Status:** engineering improvement, experimental LM mixture unpromoted. **Accepted checkpoint:** exactly the R7.7 step-14,416 full model SHA-256 `54e835b00642013d4f1bf486a8150eb1ff06c547565771bd21933e143d050f2d`. **Five inherited optimizers unchanged.** R7.11's selected 80-auxiliary-update directed writer is recovered from one 173,477-byte overlay with its new optimizer states. No separate replacement model exists.

## 1. Streaming repair

`source/r712_streaming.py` keeps a single persistent inherited rank-64 matrix and the 768-token older-history ring. At each inference step it advances the **legacy** matrix via the exact inherited `CrossWindowMirror.advance` and shifts a single token from the recent 2,048-token context into the older ring. The directed correction is recomputed from that ring **on demand** and blended into the one matrix supplied to the inherited decoder. No next-token target can enter the ring.

- At stream initialization, directed read matches the R7.11 full-window read **exactly** at nonzero gate.
- At zero gate, initial and successive stream predictions are **bitwise identical** to R7.11 and accepted R7.7 predictions.
- A nonzero-gate full-LM gradient reaches the integration gate; an independent second construction of the expected ring and recurrent matrix matches exactly through 64 steps.
- CPU **microbenchmark**, 2 threads, 2 streams, 64 *matrix read + advance* steps: 1.759 ms per two-stream batch step. This **excludes** the full network, exact-memory retrieval, tokenization, and generation overhead, and is NOT a full-token throughput measurement.
- The legacy matrix is a recurrent state; its next-state update is not equivalent to independently reinitializing the full 6,144-token history from 96 blocks. This distinction is retained intentionally.

This removes R7.11's 'nonzero gate not streamable' limitation. It does **not** mean the new reader has improved generation.

## 2. Causal sparse directed-memory LM mixture

`source/r712_lm_bridge.py` scores the 767 historical directed transitions in the 768-word older ring, combining similarity to the previous word and the R7.11 trained rank-64 directed value response. It constructs a sparse historical value distribution *without using the current target*. Only after that distribution and its confidence features exist is the actual target used to evaluate NLL. A small feature-conditioned probability-mixture gate is added to the fifth existing optimizer as **one additional parameter group**; the parent model and all existing optimizer state remain unchanged.

128 calibration updates on 256 historical training and 96 development positions selected a positive development NLL gain +0.006589 nats/word; a 96-position historical test showed +0.028898. This was **not sufficient for promotion**. In a separately reseeded 512-position audit, development correct-memory gain was +0.000261, but wrong-memory gain was **+0.007830** (correct-over-wrong disadvantage **−0.007569**). Historical test correct-memory gain +0.006849 and correct-over-wrong advantage +0.006343. Mixed-direction source controls invalidate a robust memory-specific benefit. Do not claim generalization or a complete successful LM integration.

The 10KB R7.12 gate overlay is saved only for reproducibility, and remains **unpromoted**, not activated in accepted generation. It restores with pinned parent and R7.11-delta hashes. An attempted 256-step reliability-conditioned training run on 1,024 train and 384 development positions saw no development gate passing either positive-NLL or positive correct-over-wrong controls. The run was stopped after recording update 256; **no postselection test nor new overlay** was saved from that variant. `reports/R7p12_reliability_negative.json` transparently records the incomplete terminal evaluation.

## 3. Horizon ablation: longer context alone is not better

A frozen-reader 128-query development sweep at horizons 768, 1536, 3072, 6144 older words gave CE **3.102794, 3.296192, 3.486599, 3.673287**, respectively. Top-1 **21.875%, 15.625%, 14.063%, 14.844%**. Development selected the original 768; postselection historical test of 768 had CE 3.269889 and top-1 18.75%. The same 80-update writer was trained only at 768. This is a failure of **naively widening** the frozen writer, not evidence that learned multiscale routing cannot work.

## 4. Frontier-focused promotion thresholds (future experiments)

1. **Correct-source specificity:** on several independently sampled held-out documents, corrected memory must beat wrong-source and order-shuffled memory with a book-level CI excluding zero; random-query bootstrap is not a substitute.
2. **Learned long-range retrieval:** newly trained learned multiscale/sparse reader must show positive lift at distances 2K, 8K, 32K tokens with distractors matched by key/value similarity. Compare a conventional attention/kNN memory with matched parameter and retrieval budget.
3. **Language prediction:** positive next-token NLL gain on unseen, whole-book/author-disjoint NaNROUTER books; no regression on short-context baseline; improvement on non-space and rare-word subsets.
4. **Scaling law:** measure NLL and BPB across multiple compute/data/model sizes with a fixed tokenizer, fair train-token budget, throughput, and full-context cost. Increase D/FFN/layers/experts only after the retrieval bottleneck is demonstrably learnable.
5. **Engineering:** index or segment the older stream so per-token updates are amortized rather than repeatedly embedding 768 words, and add low-memory multiscale sparse routing with a strictly causal cache.

A frontier-capable architecture is a research goal; **none of the published frontier model capability levels has been established by PAL R7.12**.

## Reproduce (two CPU threads)

```bash
PYTHONPATH=source OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python source/test_r712_streaming.py
PYTHONPATH=source OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python source/test_r712_longstream.py
PYTHONPATH=source OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python source/r712_lm_bridge.py --train 256 --dev 96 --test 96 --updates 128
PYTHONPATH=source OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python source/r712_validate.py --count 512
PYTHONPATH=source OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python source/r712_horizon_sweep.py --count 128
```

The historical PAL training/development/test splits are **not sealed**, and the original corpus carries **no per-position document IDs**. `source/fetch_nanrouter_release.py` and `source/prepare_document_corpus.py` remain available for source-disjoint book intake when pinned external corpus assets can be accessed; network access was unavailable in this container. Independent NaNROUTER new-book evaluation **has not run**.
