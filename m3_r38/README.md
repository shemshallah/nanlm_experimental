# M3 R38 — Multiscale Causal Episodic Feedback (experimental)

**Status:** built and tested locally; **not** a replacement for the independent-TEST-certified R37 result. GitHub `master` still contains R35/R32 as of 2026-11-22 and does not yet contain the R36 or R37 implementation/checkpoint. This implementation is deliberately additive, so existing M3 inference remains untouched.

## Architecture

Instead of fitting only global scalar calibration, R38 learns to exploit actual **document-local causal suffix-to-next-byte evidence**, at four nested byte suffix orders (2, 4, 8, 16). Each observed target updates counts only *after* its prediction. At every query, it forms a Dirichlet-smoothed successor distribution per suffix, using the frozen parent's probability vector for smoothing. Its 12 trainable channels express (1) suffix/parent log-density ratios, (2) disagreement between suffix scales, and (3) confidence-conditioned corrections. The 256-byte normalized readout is:

`log P_R38(b | h) = log P_parent(b | h) + theta · features(h,b) - logsumexp(…)`.

All coefficients start at **exactly zero**, guaranteeing bit-exact original predictions before training; unlike the constrained R36 readout, R38 can change the top-1 and top-5 rankings. Its causal episodic memory is bounded to 32,768 suffix addresses per document (LRU eviction) and resets between books. All stored features come from observed history, never the unobserved target.

## Recovery and lineage

`optimizer.py` requires the exact input checkpoint SHA-256, checks full Adam moments/variances/ages, and appends 12 groups without mutating the inherited parent arrays or any embedded background anchor bytes. A verified R37 continuation must load `cd7a6fa9a4ba4368b8a423ac860038dd30062eba40bcbbf2b451c8ccbd520cc4` (42 groups), creating a **54-group** checkpoint. R37 checkpoint retrieval is a precondition; **do not label a run from R32 as R37 continuation**. The repository's R32 staging adapter instead appends 12 to its verified 32 groups (44 total), using explicit separate paths/names.

## Running

`python -m unittest discover -s m3_r38 -p 'test_*.py' -v` tests exact zero parity, normalized probabilities, causality, address sensitivity, numerical gradients, reranking, 42→54 checkpoint prefix preservation, and a small synthetic online-learning run.

With SHA-verified NaNROUTER Release `corpora-short-stories-0004` extracted:

`python -m m3_r38.train_r32_staging --corpus-root /tmp/m3_r38_corpus --output-dir m3_r38/results`

This **R32 staging** run uses previously opened FIT 540–555, GATE 556–559 and DEV 596–607; R34's TEST 572–595 is excluded. Report actual matched BPB, nonspace BPB, top-1/top-5, changed rankings, per-book gains, and the unpromoted checkpoint. Results are **exploratory only**. For a true R37 successor, restore the frozen R37 source and complete 42-group checkpoint, verify its SHA, then provide a factory loading that exact parent and lock a new book-disjoint evaluation cohort and unrelated-history control **before** scoring.

## Publication rule

Do not promote this experiment on synthetic data or staging DEV. A new untouched book-disjoint TEST is required, following a successful development gate. Never use the independent TEST used to certify R37 for tuning. Preserve the accepted checkpoint, and keep a single deployed default with exact fallback modes.