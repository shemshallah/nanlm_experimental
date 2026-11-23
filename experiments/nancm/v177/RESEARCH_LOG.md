# v177 research log

1. Recovered full v176 trainable hybrid, original 128+128 signed geometry and 12k word/BPE/byte sidecars intact. Reviewed v170 `wf170_byte_next`: prequential trust is disabled only in third coder, so arbitrary prefix evaluation cannot leak future bytes.
2. Built persistent bounded prefix-DAG cache in private `wf177` handle, with row-major 256-way batched query API. C++ synthetic and native signed hybrid tests passed. No parallel hidden-state compute. Measured both raw independent prefix calls and v176's already-existing within-call sharing; do not misattribute old improvement to new code.
3. First benchmark 64 contexts/1,920 seeded samples (pre AB/BA stabilization): 562? historical `v177_cpu_batch_benchmark.json` was later overwritten. Final matched 32 contexts/512 AB/BA trials: warmed sample 1.094×. Cold 96 contexts/one sample each: 0.965×, a regression. The single-pass goal remains open.
4. Counterfactual byte-informed top-five reranking with 396 TRAIN-book examples selected a negative verifier coefficient; the fixed model tied 80→80 of 550 VAL examples and increased conditional top-five loss. Rejected.
5. Invented six-event arithmetic macro code, with escape conditioned on not completing any of five proposal strings. Derived exact mass conservation and proved through 184 synthetic and 60 trained native byte-oracle roundtrips. Observed 680 vs 682 bits on 60 words in a header-free microbenchmark, too small and quantized to infer compression improvement.
6. Repeated old 256-way byte equality on 48 vectors. Attempt at a complete 16,018-byte reference replay hit the runtime bound and produced no completed result, documented as not rerun.
7. v177 is opt-in infrastructure; default v167 open-vocab and v170 byte path protected. Archive/checkpoint/optimizer inheritance verified before cleanup; no promotion to end-to-end speed or accuracy.


## Optional two-thread experimental path: rejected

`native_parallel177.cpp` adds a distinct `wf177_parallel_*` API, using **separate private causal v170 byte-model handles** per worker and a deterministic prefix-task queue. It does perform genuine simultaneous verifier calls, unlike the default serial v177 batch. On **12** real VAL histories with two workers: **218 exact 256-way Q/P vectors**, **96 exact seeded sampling outputs**, and **12 exact five-word likelihood scores** versus v176.

The performance gate decisively rejected this design: on 32 matched VAL histories, serial prefill took **19.69 ms/context**, two-worker parallel prefill **24.90 ms/context** (0.791× throughput). Opening the two cloned model states added **6.38 s** and about **794 MB RSS**. The cloned states therefore cannot be deployed to the Android target, even though the distributions are exact. The parallel functions are opt-in and allocate no clones unless explicitly requested. **No parallel speedup claim.** The next engineering target is shared immutable weights with lightweight worker-specific mutable states.