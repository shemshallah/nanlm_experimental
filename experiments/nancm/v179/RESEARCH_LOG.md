# v179 research log — 2026-11-23

1. Audited v169/WF148/WF170: byte inference includes substantial mutable causal state. Sharing raw pointers across threads would race and violate v176 exact P. v177 two-worker full model clones were slower and memory costly.
2. Designed Linux fork/COW verifier: parent opens model once; per-worker independent causal histories after fork; no shared mutable WF148 pointer. Implemented native pipe protocol, bounded size, exact P->Q normalization using v177 original helper, speculative seeded sampler. Synthetic tests passed.
3. Initial per-prefix IPC experiment: 1/2/4 worker 24-book runs ~0.92x/0.89x/0.83x. Rejected for single-stream promotion. Corrected PSS parser to read `/proc/<pid>/smaps_rollup` by line.
4. Batched IPC (~one history/request, 32 prefixes): synthetic exactness passed. Two-worker 24-context prefill near parity; independent-book throughput slightly better. Still not a single-context win.
5. Multi-context parent-prepared verifier: verified 240 exact scores on 48 reused VAL-book contexts; four workers 1.76x with one-model PSS budget.
6. Full-context child execution: parent no longer serializes word predictor; worker performs original word prediction, top-five byte verification, seeded sample. 48-context 4-worker result 3.31257x. Separate full-48 reruns 3.36427x and 3.4789x. All exact checks passed; worker PSS increase around 3.4 MiB after workload.
7. Added /proc/self/task guard to reject unsafe fork after multithreading. Standalone native runner required for Python/Torch evaluation.
8. Final current mode is experimental; no model checkpoint or optimizer state modified; archive cumulative, deterministic, backed by independent integrity verification.
