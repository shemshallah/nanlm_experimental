# NaNCM v177 — Prefix-DAG speculative verification × exclusion-compensated macro coding

**2026-11-23. Research release, default-disabled.**  No parent weights or optimizers are changed. The original 128+128 S+/S− cloud, v175 signed temporal recurrent/copy expert, v176 normalized 12,000-word top-five+BPE+first-letter expert, and v170 256-way byte verifier remain intact. The active v167 whole-word decoder remains protected.

## Two complementary implementations

1. `native_lattice177.cpp` adds a **private** v177 model handle owning the full v176 model plus a 256-node bounded prefix cache. `wf177_begin` clears all cache state; `wf177_next` returns **exactly** the inherited v176 `Q(prefix)` and verifier `P(prefix)` 256-way vectors, memoized by literal byte prefix (no lookahead). `wf177_verify_batch` takes an array of 0–512 arbitrary causal prefixes and returns row-major `[count,256]` float Q/P tensors in one native call. `wf177_prepare_top5` factorizes five whole-word proposal paths into a shared prefix graph and returns original-v170 normalized average log likelihood. `wf177_sample` uses the same seeded `draw Q → accept min(1,P/Q) → sample residual(P−Q)+` law as v176 and reuses cached distributions. The underlying native oracle is still called **serially**, not in parallel.

2. `macro_code177.py` adds a **lossless six-symbol third coder** (five word events + escape). Its candidate events are probabilities of full exact byte strings under the **byte verifier P**, not the word model's 12,000-way scores. For prefix `s`, let `M(s)` be the probability mass of complete proposed words that start with `s`. The conditional escape distribution is

   `P(b|s,escape) = [P(s)·P(b|s) − M(sb)] / [P(s) − M(s)]`.

   This produces an **exact refactorization** of the original P law in real arithmetic. An in-top-five word is coded as one event; any other byte string uses the escape symbol followed by compensated conditional byte events. Finite precision arithmetic coding uses `range_code176.py`. It is expected to have the **same Shannon entropy** as P, not magically better bits-per-byte. Expected number of arithmetic operations can improve for frequent complete words, but computing event masses requires prior byte-verifier work.

## C API (source-only summary)

`wf177_open_full(teacher, model, geometry, cloud, bytepack, escape, crosslevel, bpe)` returns a private v177 handle. `wf177_begin` must be used before any other call or for a new history. `wf177_next`, `wf177_top5`, `wf177_prepare_top5`, `wf177_verify_batch`, `wf177_stats`, `wf177_sample`, and `wf177_close` are opt-in. No v176/v175/v170 old handle is exposed for mutation.

Batch shape: `wf177_verify_batch(handle, bytes, bytes_len, offsets_uint32, lengths_uint32, count, out_q_float32, out_p_float32)`, with count ≤512, prefix bytes ≤40 each and output size `256×count` for each array. Inputs are validated for bounds and cannot mutate model weights.

## Measured results, interpretation and important limits

- Native C++ synthetic test: all five proposal words, full 256-way normalization, exact 255-seed seeded sampler equivalence, cache invalidation, malformed inputs and duplicate prefix batch parity pass.
- Trained parity on **50 validation contexts**: **1,219 byte-for-byte identical** 256-way P/Q vector pairs; **500 identical seeded sample outputs**; **250 matching top-five rankings**; 50 cache resets; no weight changes.
- Warm workload, 32 VAL contexts ×16 drafts/context, **512 identical seeded outputs**. Independent top-five verification needs 562 original prefix calls, while both v176's within-call prefix sharing and v177 need **400 unique prefixes** (28.83% fewer). Once warmed, v177's persistent cache carries across repeated calls: all second-pass top-five verifications and grouped prefixes used zero additional oracle calls. In this workload the warm v177 sampler ran **1.094× as fast** as v176 sampler (2.712 vs 2.966 ms/draft, matched AB/BA order). First-pass new-cache verifier is *not* faster than a warmed v176 verifier.
- Cold workload, one draft per history including begin, **96 identical outputs**: v177 relative throughput **0.965× v176**. That is ~3.5% **slower**, demonstrating no general streaming acceleration. Neither workload has parallel oracle execution or measured total application tokens/s.
- `macro_code177.py` tested on 184 synthetic words and `test_real_macro177.py` on **60 actually trained validation histories** (16 true words were within proposed top-five): every case decoded losslessly and the ideal macro probability matched the original byte P. Plain arithmetic coding emitted 682 bits versus macro coding's 680 bits over the 60 samples; that 2-bit difference is quantization/termination, **not** an entropy or full-text BPB improvement. The six-event factorization requires P for all five proposed strings and is opt-in research.
- Byte-likelihood reranker calibrated on 396 TRAIN-book examples (top-one 63→78) **failed to generalize** on 550 examples across all 55 VAL books (80→80), and its conditional top-five NLL worsened from 1.2975 to 1.3293. Rejected.
- Exact old native byte ABI parity: 48 full normalized 256-way vectors on previously explored DEV/TEST, maximum error 0. Old v176 16,018-byte result is inherited, **not newly reexecuted to completion** this round.

**Promotion:** the shared-prefix verifier and macro coding are retained as opt-in infrastructure, not an active accuracy or frontier-speed promotion. No new trained neural checkpoint exists in v177; all v176/v175/v174/v173 optimizer and model checkpoints are retained verbatim in the cumulative recovery ZIP. For true single-pass acceleration, a parallel/state-incremental byte verifier with independent causal state and bounded memory is required.

## Reproduce

```
make v177
make v177-source-smoke
make test-v177
python experiments/nancm/v177/bench_batch177.py --n 32 --samples 16
python experiments/nancm/v177/bench_stream177.py --n 96
python experiments/nancm/v177/eval_rerank177.py --train 396 --val 550
```

Full native runs require all inherited trained sidecars in the cumulative recovery ZIP. GitHub is source-only.


## Optional two-thread experimental path: rejected

`native_parallel177.cpp` adds a distinct `wf177_parallel_*` API, using **separate private causal v170 byte-model handles** per worker and a deterministic prefix-task queue. It does perform genuine simultaneous verifier calls, unlike the default serial v177 batch. On **12** real VAL histories with two workers: **218 exact 256-way Q/P vectors**, **96 exact seeded sampling outputs**, and **12 exact five-word likelihood scores** versus v176.

The performance gate decisively rejected this design: on 32 matched VAL histories, serial prefill took **19.69 ms/context**, two-worker parallel prefill **24.90 ms/context** (0.791× throughput). Opening the two cloned model states added **6.38 s** and about **794 MB RSS**. The cloned states therefore cannot be deployed to the Android target, even though the distributions are exact. The parallel functions are opt-in and allocate no clones unless explicitly requested. **No parallel speedup claim.** The next engineering target is shared immutable weights with lightweight worker-specific mutable states.