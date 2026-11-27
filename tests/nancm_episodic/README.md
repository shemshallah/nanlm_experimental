# v54 causal episodic continuation
Verified v53 improves v50 across all three held-out cohorts (1728 words):
v50 246/479/837 top1/5/32 -> v53 265/499/862. Reference-only contribution
small; local-only NLL 6.332656/6.756478/6.740495, full 6.331588/6.755975/6.740330.
v53 chosen as component parent; v50 field and v53 trained credit/optimizer
artifacts immutable, no reset. Add three causal donors: variable suffix
depth up to 8, skip lag agreement, recency-weighted topic copy. All use
already observed words only. 18 new trainable gate parameters per arm,
256 updates each, secondary unindexed fit books rather than v53's primary.
Four matched arms (three single donors and full); calibration includes exact
zero-strength rollback. Wrong-context swaps older history while retaining
last four words and parent unchanged; this creates an artificial splice
and is diagnostic only. Evaluate same DEV IDs and samples as v53.
Require exact parent top1/5/32 counts and NLL within 1e-6 before marking
integrated result final. Local test executed in container: constructed
128 train,64 held-out,128 updates; NLL 1.9408360559467925 -> .08813029424818439,
FD error 2.0462992411651726e-11, mass error zero, disabled exact,
deeper-context disambiguation. Not actual language accuracy.
The new training artifact references the immutable v53/v50 files; their
optimizer moments remain in those artifacts. Still categorical law only,
not raw BPB or production promotion. Actual corpus run pending.
