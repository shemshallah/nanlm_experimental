# v51 cross-fitted cooperative field learning
Exact leave-one-reference-document-out subtraction of unigram and 1/2/3-gram
counts. Same smoothing and interpolation as v48/v50; no target features.
Local executed check (generated documents, not language): 385 queries,
rebuild vs subtraction error 0, feature error 0, scalar vs dense error 0.
13 gate finite differences max 7.543934921194762e-11, field gradient norm
0.040044380890983895, zero gate adjustment exact.
Actual corpus audit launches without checkpoint recovery: at most 24,576
reference positions; compare contaminated indexed vs excluded target NLL,
unindexed calibration count NLL, and scalar/dense agreement. The reference
codebook still uses reference documents (normal supervised training); only
continuation counts and unigram are excluded from that document's mixture.
Manual workflow dispatch after successful v50 run 45914210629 resumes
the exact coupled Field and its AdamW moments (768 -> 1024 updates per
matched fixed/adaptive arm). A new 13-parameter gate uses target-free
support, diversity, max-frequency and trust features of the three
context orders. Both arms optimize the deployed normalized mixture NLL,
with identical initial model, count priors, samples and budget. Training
gate center clamped to [.125,.875]; calibration includes zero/one and
zero adaptation. Wrong-context field control keeps count context and gate
features correct. Same DEV samples as v50. Immutable v50 artifact SHA,
matching codebook and manifest required. Not a replacement for v37
production lineage; v48 development baseline unchanged. No raw byte/BPB
claim: categorical word+unknown law, awaiting v49 open-vocabulary bridge.
This turn the integrated runner has not been executed; the audit runs on
push, and continuation requires explicit dispatch after complete v50.
