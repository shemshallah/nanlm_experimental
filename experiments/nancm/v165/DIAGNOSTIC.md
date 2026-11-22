# NaNCM v165 — Candidate-support audit on independent DEV books

This audit runs the frozen native v164 whole-word distribution (`wf164_candidate_count` and `wf164_candidates`) on exactly 480 deterministically spaced word positions from 24 distinct held-out DEV books. Each call receives only earlier bytes; no target suffix enters `wf148_begin`.

| Diagnostic | Count |
|---|---:|
| DEV word positions | 480 |
| Distributional memory active | 203 |
| Correct next word represented in its candidate set | 70 |
| Conditional distributional candidate recall | 70/480 = 14.583% |
| Average number of distributional candidates per position (including inactive) | 4.627 |

**This is a branch-specific recall ceiling, not a bound on the full v164 or v165 decoder.** The parent v162 component, open-vocabulary speller, and other generators may propose words outside the v164 distributional memory.

The observed 14.583% branch-support ceiling explains why reweighting the existing 16-word memory proposal cannot, by itself, reach 80% strict top-1. v165's trained S+/S− edge reranker made worse held-out DEV spelling/EOS loss at every tested strength, despite learning a TRAIN-index conditional objective.

Proposed architecture pivot: learn a causal world-state-driven open-vocabulary generator jointly with the original signed geometric cloud, then train/evaluate using *actual native complete-word prediction loss* on book-disjoint data. A larger reranking budget is not a substitute for support recall.

Reproduce from complete v165 recovery archive:

```sh
OPENBLAS_NUM_THREADS=1 python scripts/diagnose_recall_ceiling165.py
```

Machine-readable evidence: `results/v165_candidate_coverage_ceiling.json`. Original v164 and inherited models and optimizers are unchanged. v165 remains unpromoted.
