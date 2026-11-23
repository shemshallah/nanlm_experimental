# NaNCM v178 Claim ledger — 2026-11-23

| Claim | Status | Evidence / qualification |
|---|---|---|
| Shared single inherited v176 model for many independent sessions | TESTED | 32 live sessions, one model instance, real sidecars |
| Exact 256-way P/Q rollback relative to v176 | PASS | 1,960 complete vector pairs, no differences, 32 distinct VAL books |
| Exact seeded speculative acceptance | PASS | 256/256 seeded samples match v176 |
| Top-five whole-word probabilities preserved | PASS | 320/320 native rank, spelling, float-probability entries |
| Source-only concurrent interleaved smoke | PASS | GitHub Actions #45250088322 |
| Eight/sixteen/thirty-two independent causal histories | PASS | 64 alternating visits, one shared model |
| Incremental resident memory for 32 sessions | MEASURED | 180,224 bytes in one process; not a portable bound; model base already resident |
| Cold verifier speedup | NOT ESTABLISHED | 32 contexts 20.35 vs 20.11 ms/context incl session/begin; ~1% difference may be noise |
| Warm prefix-repetition speedup | MICROBENCH ONLY | ~19.97× on identical prefix queries, not application decode throughput |
| True parallel neural inference across contexts | NOT IMPLEMENTED | Hub mutex protects mutable WF148 through WF170 oracle |
| Accuracy / BPB improvement over v176 | NONE CLAIMED | Byte law and trained parameters frozen |
| Open-book generalization | NOT TESTED | VAL repeatedly explored; no new pristine book-disjoint TEST |
| Deployment promotion | NOT YET | Opt-in v178 only; require end-to-end throughput, memory and fresh test |
