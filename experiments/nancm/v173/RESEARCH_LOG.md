# NaNCM v173 research log — 2026-11-22

1. Recovered and verified the v172 689-entry cumulative checkpoint. Preserved 128+128 signed cloud, 32D lexical head, 16D ANN, native v170 byte stream.
2. Audited v169 portfolio: inherited parent predicts the word "the" 302/850 TRAIN and 79/244 DEV contexts, while the actual next word is "the" 50/850 and 10/244 respectively. Candidate bias is large, but naive frequency corrections do not generalize.
3. Ran TRAIN-only book-calibrated prior correction, causal 1–3gram memory, and native byte-word agreement; rejected all as strict-word upgrades after DEV results.
4. Resumed v172 step 6,000 AdamW model and trained 12k exact CE; 1,536-context TRAIN SHA validation NLL best moved from 6.263737 to 6.260751 by step 6,880, with 3 optimizer groups/5 moments preserved. The DEV teacher standalone word NLL 6.528180→6.525256 but accuracy 16→14/227.
5. Packed v173 exact CE sidecar in fully validated N172TCH1 fingerprinted format, checked native probability parity on 21 historical DEV contexts.
6. Built native C++ whole-word first-letter product with prequential byte sidecar, exact EOS and 26-letter simplex normalization, opt-in gamma0.1. 240 exact v172 disabled rollback, 48 active normalized identities and 144 exact nonroot vectors passed.
7. Matched full strict 96-beam top1 trained calibration 23→24/197; exposed DEV 27→27/244, rejected promotion. First-letter NLL improves from 2.9291496 to 2.8955784 but accuracy does not.
8. Replayed complete 16,018 native byte predictions on 48 book windows, identical to v170/previously reused TEST, with 292 exact legacy rollback vectors. Source-only GitHub master commit, CI, ZIP SHA, Library one-archive policy tracked in release manifest.