# NaNCM fresh structural foundation
No old checkpoints imported and no gradient training. Python research reference, not production or native C.
## Real NaNROUTER data
The workflow fetches and checksum-verifies the Google English 10k lexical release, Gutenberg short stories, sentences and paragraphs batch 1, WikiText-103 validation and chemistry. Reference index: 24 Gutenberg books (71,351 normalized words). Gutenberg IDs are partitioned consistently across representations. Diagnostic windows: 360 word targets per category; sentences and paragraphs share some held-out IDs, so the pooled result is descriptive, not independent. WikiText and chemistry are cross-domain probes. Split 9 is reserved, not used. The wordlist is lexical structure, never ordered training text.
## Completed initial experiments
- v0 commit 301d143: run 45878294739; retrieval vs signed lexical feedback, disjoint spelling escape, exact prefix probabilities. Results in results/v0.json.
- v1 commit 236fbd0: run 45878428002; explicit 8D sparse signed graph, all three pair classes, forward/backward feedback. Results in results/v1.json.
- v2 commit 6fded08: run 45878569621; hierarchical mass and raw 256-byte projection.
- v3 commit 36131a6: run 45878695544; fixed-graph contraction, deterministic rollback, zero-initialized persistent signed gains and directional ablations. Check run status/metrics before promotion.
## Current architecture
- Causal local, skip, discourse and per-document episodic retrieval.
- 128 lower and 128 upper points, C(256,2)=32640 unordered cells, 8 coordinates each in each of two signed channels (522240 persistent gains).
- Sparse context-derived evidence, separately modulated S+/S- gains; no pretrained weights. Bounded fixed-graph recurrence with row-sum normalization; outer candidate expansion capped.
- Candidate distribution plus disjoint spelling escape, exact prefix marginalization and bounded best-first whole-word ranking with explicit certification.
- 256-byte output via case/letter/EOS projection and raw causal byte counts; non-ASCII bytes retained but Unicode words are not modeled as word identities.
- Fixed graph contraction proof: normalized symmetric W has operator norm <=1, L2 ball projection is nonexpansive, update Lipschitz <=0.35. This is not a convergence guarantee for the outer retrieval loop.
## Open gates
Do not promote or start full training yet: complete real-data v2/v3 measurements; audit more random windows/book IDs and avoid aggregate double-counting overlapping representations; compare graph and sign ablations; improve speed; add full differentiable training and optimizer checkpoint/resume; extend semantic/Unicode word handling and BPE metrics. Word spelling bits/symbol is not raw byte BPB. No frontier accuracy claim.
Run locally with data extracted as the workflow expects: python nancm_structural/base.py --data data --out metrics.json.

## Continued structural work (v4–v6)
- v4 (11730f7, run 45880949496): full support probability mixture; discounted causal counts; distributional role analogy, sparse signed counterfactual readout, known-word mass preserved outside the dynamic shortlist. Matched 1,800 word positions plus 18 fresh held-out Gutenberg IDs (1,728 targets), unique across three representations. Raw byte probe and correct/unrelated history test. This is a new probability law, not merely a larger shortlist. The conditional-support metric is not directly comparable to earlier truncated candidate recall; use top-k and shortlist recall explicitly.
- v5 (84297f5, run 45881077132): fixed-degree symmetric contraction, analytic implicit adjoint for the actual signed word-loss readout. Finite differences temporarily perturb real addressed gains; the original checkpoint must be byte-identical afterward. Bayesian raw/word route is latched at each ASCII segment and updated only after observed bytes. Tests compare segment probability to its two generative route laws. No optimizer steps.
- v6 (d7209cb, run 45881181818): C++17 sparse forward/adjoint solver exposed by a C ABI, with invalid-input checks, convergence checks, Python prediction parity and real-corpus finite-difference tests. Python still owns graph construction and retrieval. Native speedup must be measured, not assumed.
- v2 and v3 completed: v3 fixed graph converged 5982/5982, all 1800 word ranks certified; word top1 17.11%, top5 29.00%, raw BPB 2.7574. Hierarchical raw BPB 2.7539 and retrieval word top1 18.17% remain stronger on their respective metrics. v3 not promoted. Complete logs archived in results/v2.json and results/v3.json.
- v4/v5/v6 runs were not yet complete at the time of this entry. Do not infer predictive improvements or parity/gradient success until their logs are retrieved. These are structural experiments, not replacement production checkpoints.
