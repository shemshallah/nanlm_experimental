# v177 provenance, frozen ancestry, and measurement rules

Parent: verified `NaNCM_v176_SIGNED_BPE_WORDTOP5_SPECULATIVE_THIRDCODE_RECOVERY_20261123.zip`, SHA-256 `c49b8feb6159bb38e070ddc45331a457b7ab968d30d72870862c3034ed5a6005`, all 918 entries preserved (except intentionally extended root Makefile/README with byte-exact copies in ancestry).

Training sources: inherited 396 TRAIN, 55 VAL, 49 previously exposed TEST books from NaNROUTER Release `corpora-short-stories-0003` and saved code/sidecars. All v177 evaluation examples are drawn from the **already known 55 VAL books**, never pristine; reranking alpha selected only on TRAIN-book examples, then evaluated on VAL. No new optimizer or parameters trained for v177.

Native test `test_native177.py` verifies literal byte-exact P/Q vectors vs frozen v176 for identical causal histories. Seeded stochastic sampling uses the same `std::mt19937_64` draws. `test_real_macro177.py` uses the exact inherited native P at every prefix; original target words are only used for scoring/roundtrip, never prefix prediction. Synthetic uniform/nonuniform oracles are separate from empirical model checks. Clock times are CPU wall timings, AB/BA order alternated for warm and cold benchmarks; not a claim of parallelism.

`macro_code177.py` deliberately trades one exact macro arithmetic coding event for exact exclusion-conditioned byte escape. It does not alter the original byte probability law in real arithmetic. Finite precision length differences and header omission must not be confused with true corpus compression gains.

Archive manifest enumerates each retained file, SHA-256; parent original root Makefile and README are preserved separately. Only release ZIP is to be kept in ChatGPT Library after successor upload/readback verification; other independent architecture archives untouched.


## Optional two-thread experimental path: rejected

`native_parallel177.cpp` adds a distinct `wf177_parallel_*` API, using **separate private causal v170 byte-model handles** per worker and a deterministic prefix-task queue. It does perform genuine simultaneous verifier calls, unlike the default serial v177 batch. On **12** real VAL histories with two workers: **218 exact 256-way Q/P vectors**, **96 exact seeded sampling outputs**, and **12 exact five-word likelihood scores** versus v176.

The performance gate decisively rejected this design: on 32 matched VAL histories, serial prefill took **19.69 ms/context**, two-worker parallel prefill **24.90 ms/context** (0.791× throughput). Opening the two cloned model states added **6.38 s** and about **794 MB RSS**. The cloned states therefore cannot be deployed to the Android target, even though the distributions are exact. The parallel functions are opt-in and allocate no clones unless explicitly requested. **No parallel speedup claim.** The next engineering target is shared immutable weights with lightweight worker-specific mutable states.