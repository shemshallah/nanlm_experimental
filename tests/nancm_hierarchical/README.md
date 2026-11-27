# v50 signed hierarchical lexical field
v48 design baseline retained, v37 immutable. This component uses the same
384-reference-book index and 49,960-word corpus+wordlist setup as v48.
It learns from up to 192 distinct reference positions per book (at most
73,728 unique contexts), with 768 updates x 64 exposures per arm (49,152
sampled exposures). Unlike the contaminated v43 readout, its fitting
inputs contain only preceding words and reference-trained lexical features,
NOT any indexed continuation counts or target-based candidate retrieval.
The reference index is used only as an unchanged held-out baseline and
calibration mixture. Two independently trained identical-initialization
arms differ solely in signed cross-rail recurrence. Each rail has bounded
low-rank support/opposition maps in 128D, tied learned lexical identity,
and two cross-rail recurrence passes. Signed geometry classes contain at
most 128 words; the sparse target likelihood is exactly the dense law.
Executed local float64 preflight (37 vocabulary, 24 groups): target log
probability error 0, sum error 4.440892098500626e-16, 13 nonzero gradient
groups, maximum finite-difference error 1.9931386947792326e-09, minimum
gradient norm 1.1706300441666762e-05, exact zero-cross parity.
These are numerical tests, not language accuracy. Actual workflow trains
both arms on identical samples and tests count, dense field, calibrated
mixtures, unrelated history and forced .5 mixing, on v48's held-out
sample seeds/IDs. Reports word top1/5/32 (OOV always missed) and categorical
NLL including aggregate unknown class, NOT raw BPB or open-vocabulary
word likelihood. The v49 open word/event bridge is needed for integration.
This is a separate component diagnostic, not an alternate production
checkpoint or inherited v37 optimizer reset. Corpus integration pending.
