# NaNCM v178 — shared verifier hub, real trained parity

Research opt-in (default disabled), 2026-11-23. The single C++ model instance is shared by multiple independently resettable causal sessions. Each session stores its own history, top-five word snapshot and 256-entry prefix P/Q cache. The mutex protects oracle misses, so true parallel neural inference is not implemented. Exact inherited v176 P/Q and sampling law are preserved.

See CLAIM_LEDGER.md, RESEARCH_LOG.md, PROVENANCE.md, RESULTS.json, test_trained178.py and bench_cold178.py.

Real native tests using unmodified trained model sidecars recovered from verified v177 parent ZIP: 32 distinct VAL books, 64 alternating session visits, 1,960 exact 256-way P/Q pairs, 256 identical seeded samples, 320 identical word top-five entries, one model instance shared by 32 sessions (180,224 bytes incremental resident memory for sessions). 32-context first-pass microbenchmark incl begin 20.349ms v176 vs 20.105ms v178; not a reliable speed win. Repeated identical prefix queries: 0.3358ms vs 0.01682ms per context; warm cache speeds query reuse but not demonstrated full decoding throughput. No new weights or optimizers.

Build `make v178`; execute `make v178-source-smoke` for source CI and `python experiments/nancm/v178/test_trained178.py --n 32` / `python experiments/nancm/v178/bench_cold178.py --n 32` for checkpoint gates. GitHub master is source only. CI source pass run #45250088322. No claimed frontier accuracy improvement.
