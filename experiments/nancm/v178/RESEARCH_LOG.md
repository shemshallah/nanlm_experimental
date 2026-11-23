# NaNCM v178 research log

2026-11-23

1. Audited v177: two private verifier workers clone the WF170/WF148 signed-cloud model stack; v177 two-worker CPU throughput 0.791× baseline and ~794 MB RSS extra. True shared table/state separation is not available without an invasive refactor because the nested model hierarchy owns mutable causal state.
2. Introduced a refcounted shared model hub, with a mutex for miss-only oracle calls; sessions own history, top-five snapshots, and 256-node bounded exact-P/Q caches. On switching to an uncached context, restore its correct history by re-running inherited v176 begin.
3. CI source smoke initially failed from a duplicate synthetic test main(); corrected to include v176 stubs directly.
4. CI then found warm-cache invariant violated after 300 random speculative prefixes reached 256-node cap; fixed test to reset bounded cache before asserting second-pass zero oracle calls. Successful Actions run #45250088322.
5. Recovered v177 ZIP (235,608,878 bytes), SHA-256 501a360d31be5e63f9da7e480432f640cbbae9e6b5e3b9e0cee39bc76ced5572, 963 entries, CRC verified. Rebuilt v178 native C++ from exact GitHub source (git blob 29e2c8e87b940408986984edc2dbe4b488757e26).
6. Trained sidecar check, 32 distinct validation books, 64 alternating visits: 1,960/1,960 exact P/Q float-array comparisons; 256/256 seeded byte sequences with matching acceptance/rejection; 320/320 exact top-five word entries; one model instance for 32 sessions; additional session RSS 180,224 bytes in the loaded process.
7. Fairer first-pass benchmark, 32 contexts, each distinct prefix exactly once: 20.349ms/context v176 vs 20.105ms/context v178, including begin/session open; first-pass ratio 1.012, no reliable speedup. Warm repetition 0.336ms vs 0.0168ms/context; 19.97× query-level speedup, not end-to-end generation.
8. No inherited optimizer, weights, or model sidecar changed. No speculative sampling-law change. Default remains protected at v177/v176, shared hub is research/opt-in.

Future route: detach immutable WF148 and byte n-gram tables into refcounted arena, move WF148 conditional caches to a cheap per-session state, then permit truly parallel missed queries without critical-section serialization. Require precise multi-book P/Q parity, Android RSS, and real cold completion throughput before promotion.
